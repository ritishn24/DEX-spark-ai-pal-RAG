import os
import uuid
import base64
import requests
import tempfile
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Load environment variables
load_dotenv()
 
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# Utils
from utils.auth import AuthManager
from utils.database import DatabaseManager
from utils.document_processor import DocumentProcessor
from utils.image_processor import ImageProcessor
from utils.vector_store import VectorStore
from utils.codebase_processor import CodebaseProcessor
from utils.rate_limiter import rate_limit
from utils.file_validator import FileValidator
 
auth_manager = AuthManager()
db_manager = DatabaseManager()
doc_processor = DocumentProcessor()
img_processor = ImageProcessor()
vector_store = VectorStore()
codebase_processor = CodebaseProcessor()
file_validator = FileValidator()
 
# ===============================
# ✅ LLM Client (Vision-capable)
# ===============================
# ===============================
# ✅ LLM Client (Vision-capable)
# ===============================
class LLMClient:
    def __init__(self):
        self.llm_server_url = os.getenv('LLM_SERVER_URL', 'http://localhost:8000/v1/chat/completions')
        self.llm_model_path = os.getenv('LLM_MODEL_PATH', '/root/.cache/huggingface/')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        logger.info(f"LLM Client initialized - OpenAI: {'Available' if self.openai_client else 'Not Available'}")
        logger.info(f"Local LLM URL: {self.llm_server_url}")

    def generate_response(self, user_message, context="", image_url=None, max_tokens=1000, temperature=0.1, system_context=""):
        logger.info(f"Generating response for message: {user_message[:100]}...")
        logger.info(f"Context length: {len(context)}")
        
        try:
            local_response = self._call_local_llm(user_message, context, image_url, max_tokens, temperature, system_context)
            if local_response:
                logger.info("✅ Local LLM response successful")
                return local_response
        except Exception as e:
            logger.error(f"[Local LLM Error] {e}")

        # Fallback to OpenAI if local LLM fails or is unavailable
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client is not initialized")
            response = self._call_openai_gpt4o(user_message, context, image_url, max_tokens, temperature, system_context)
            logger.info("✅ OpenAI GPT-4o response successful")
            return response
        except Exception as e:
            logger.error(f"[OpenAI GPT-4o Error] {e}")

        logger.error("❌ All LLM options failed")
        return "I'm sorry, I'm currently unable to process your request. Please try again later."
    
    def _call_local_llm(self, user_message, context="", image_url=None, max_tokens=1000, temperature=0.1, system_context=""):
        system_prompt = system_context or "You are a helpful AI assistant. Provide accurate and helpful responses based on the context provided."

        # Your local server payload doesn't support complex messages or image URLs.
        # It expects a string for content. We'll simplify the message content.
        message_content = self._format_message_with_context(user_message, context)
        
        # The user's provided payload uses a file path as the model name.
        # We should use this path instead of a simple model name.
        model_name = self.llm_model_path
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Check if an image_url is provided. If so, and the local LLM doesn't
        # support it, you might want to raise an error or handle it gracefully.
        # For simplicity, we will skip the image for the local call as your
        # provided payload doesn't show support for it.
        if image_url:
            logger.warning("Image URLs are not supported by the current local LLM payload format. Skipping image analysis.")
            
        try:
            response = requests.post(self.llm_server_url, json=payload, timeout=60) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"[Local LLM] Request failed: {e}")
            return None

    def _call_openai_gpt4o(self, user_message, context="", image_url=None, max_tokens=1000, temperature=0.1, system_context=""):
        # This method is already configured correctly and can be left as is.
        system_prompt = system_context or "You are a helpful AI assistant. Provide accurate and helpful responses based on the context provided."

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt)
        ]

        if image_url:
            messages.append(ChatCompletionUserMessageParam(
                role="user",
                content=[
                    {"type": "text", "text": self._format_message_with_context(user_message, context)},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            ))
        else:
            messages.append(ChatCompletionUserMessageParam(
                role="user",
                content=self._format_message_with_context(user_message, context)
            ))

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Note: GPT-4o is a better choice for images
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def _format_message_with_context(self, user_message, context=""):
        return f"Context:\n{context}\n\nUser Question: {user_message}" if context.strip() else user_message
    
    def test_connections(self):
        # This method is already configured correctly.
        status = {"local": False, "openai": False}
        
        # Test local LLM
        try:
            test_response = self._call_local_llm("Hello", max_tokens=10)
            status["local"] = bool(test_response)
        except:
            pass
            
        # Test OpenAI
        try:
            if self.openai_client:
                test_response = self._call_openai_gpt4o("Hello", max_tokens=10)
                status["openai"] = bool(test_response)
        except:
            pass
            
        return status
 
llm_client = LLMClient()
 
# ===============================
# ✅ Flask Routes
# ===============================
 
@app.route('/')
def index():
    return redirect(url_for('chat')) if 'user_id' in session else redirect(url_for('login'))
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')
 
        if action == 'login':
            user = auth_manager.authenticate_user(username, password)
            if user:
                session.update({
                    'user_id': user['id'],
                    'username': user['username'],
                    'session_id': f"sess_{str(uuid.uuid4()).replace('-', '')}"
                })
                flash('Login successful!', 'success')
                return redirect(url_for('chat'))
            else:
                flash('Invalid credentials', 'error')
        elif action == 'register':
            email = request.form.get('email')
            if auth_manager.create_user(username, password, email):
                flash('Registration successful!', 'success')
            else:
                flash('Username already exists', 'error')
    return render_template('login.html')
 
@app.route('/chat')
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    chat_history = db_manager.get_chat_history(session['user_id'])
    return render_template('chat.html', username=session['username'], chat_history=chat_history)
 
@app.route('/send_message', methods=['POST'])
@rate_limit
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        chat_mode = data.get('mode', 'general')  # general, codebase, image
        
        logger.info(f"Processing message in {chat_mode} mode: {user_message[:50]}...")
 
        db_manager.save_message(session['user_id'], session['session_id'], 'user', user_message)
 
        history = db_manager.get_session_messages(session['user_id'], session['session_id'])
        memory_context = "\n".join([f"{m['role'].capitalize()}: {m['message']}" for m in history[-10:]])
 
        vector_context = ""
        system_context = _get_system_context(chat_mode)
        
        # Different context retrieval based on chat mode
        if chat_mode == 'rag':
            # RAG mode - search documents only
            if vector_store.collection_exists(session['session_id']):
                logger.info("Searching document collection for RAG...")
                relevant_docs = vector_store.search_documents(session['session_id'], user_message)
                vector_context = "\n".join([doc.get('text', '') for doc in relevant_docs])
                logger.info(f"Found {len(relevant_docs)} relevant document chunks for RAG")
            else:
                logger.warning("No document collection found for RAG")
                vector_context = "No documents have been uploaded yet. Please upload documents to use RAG functionality."
        elif chat_mode == 'codebase':
            # Codebase mode - search code collections
            if vector_store.collection_exists(f"code_{session['session_id']}"):
                logger.info("Searching codebase collection...")
                relevant_docs = vector_store.search_codebase(f"code_{session['session_id']}", user_message)
                vector_context = _format_code_context(relevant_docs)
                logger.info(f"Found {len(relevant_docs)} relevant code chunks")
            else:
                logger.warning("No codebase collection found")
                vector_context = "No codebase has been uploaded yet. Please upload a ZIP file or provide a GitHub URL."
        else:
            # General chat mode - no document context
            vector_context = ""
 
        full_context = f"""Chat History:
{memory_context.strip()}
 
Relevant Docs:
{vector_context.strip()}"""
        
        logger.info(f"Full context length: {len(full_context)}")
        ai_response = llm_client.generate_response(user_message, full_context, system_context=system_context)
 
        db_manager.save_message(session['user_id'], session['session_id'], 'assistant', ai_response)
 
        return jsonify({
            'user_message': user_message,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[send_message] Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to process message'}), 500
 
def _format_code_context(relevant_docs):
    """Format code context for better LLM understanding"""
    context_parts = []
    for doc in relevant_docs:
        if doc.get('chunk_type') == 'structure':
            context_parts.append(f"Project Structure:\n{doc.get('content', '')}")
        else:
            file_path = doc.get('file_path', 'unknown')
            language = doc.get('language', 'text')
            content = doc.get('content', '')
            context_parts.append(f"File: {file_path} ({language})\n```{language}\n{content}\n```")
    return "\n\n".join(context_parts)

def _get_system_context(mode):
    """Get system context based on chat mode"""
    if mode == 'rag':
        return """You are a RAG (Retrieval-Augmented Generation) assistant. Your primary role is to answer questions based on the uploaded documents provided in the context. 
        
        IMPORTANT INSTRUCTIONS:
        1. Always prioritize information from the provided document context
        2. If the context contains relevant information, use it to answer the question
        3. If the context doesn't contain relevant information, clearly state that the information is not available in the uploaded documents
        4. Be specific about which document or section you're referencing when possible
        5. Provide accurate, context-based responses and avoid making up information not present in the documents
        
        Format your responses clearly and cite the source material when relevant."""
    elif mode == 'codebase':
        return """You are a code analysis expert. Help users understand codebases, explain functionality, 
        suggest improvements, and answer questions about code structure and implementation. 
        Provide clear explanations with code examples when relevant."""
    elif mode == 'image':
        return """You are an image analysis expert. Describe images in detail, identify objects, 
        people, text, and provide contextual information about what you see."""
    else:
        return """You are a helpful AI assistant. Provide accurate, clear, and well-structured answers."""

@app.route('/upload_file', methods=['POST'])
@rate_limit
def upload_file():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        file = request.files.get('file')
        upload_type = request.form.get('type', 'document')  # document, image, codebase
        
        if not file or not file.filename:
            return jsonify({'error': 'No file uploaded'}), 400
            
        logger.info(f"Processing file upload: {file.filename}, type: {upload_type}")
 
        filename = secure_filename(file.filename or "uploaded_file")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Validate file
        is_valid, error_msg, file_info = file_validator.validate_file(file_path, filename)
        if not is_valid:
            os.remove(file_path)
            logger.error(f"File validation failed: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        logger.info(f"File validated successfully: {file_info}")
 
        result = {}
 
        if upload_type == 'codebase' and filename.lower().endswith('.zip'):
            # Process codebase from ZIP
            logger.info("Processing codebase from ZIP...")
            processed_chunks = codebase_processor.process_codebase(file_path, 'zip')
            vector_store.add_codebase(f"code_{session['session_id']}", processed_chunks, filename)
            logger.info(f"Added {len(processed_chunks)} codebase chunks")
            result = {
                'type': 'codebase',
                'filename': filename,
                'message': f'Codebase "{filename}" processed successfully!',
                'chunks': len(processed_chunks),
                'files_processed': len([c for c in processed_chunks if c.get('chunk_type') == 'code'])
            }
        elif file_info['extension'] in ['docx', 'pdf', 'txt', 'md', 'doc']:
            # Process document
            logger.info(f"Processing document: {file_info['extension']}")
            processed_text = doc_processor.process_document(file_path, file_info['extension'])
            
            if not processed_text:
                logger.error("No text extracted from document")
                return jsonify({'error': 'Could not extract text from document'}), 400
                
            vector_store.add_documents(session['session_id'], processed_text, filename)
            logger.info(f"Added {len(processed_text)} document chunks")
            result = {
                'type': 'document',
                'filename': filename,
                'message': f'Document "{filename}" processed successfully!',
                'chunks': len(processed_text)
            }
        elif file_info['extension'] in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
            # Process image
            logger.info("Processing image...")
            ai_response = img_processor.analyze_with_ai(file_path)
            result = {
                'type': 'image',
                'filename': filename,
                'description': ai_response,
                'message': f'Image "{filename}" analyzed successfully!'
            }
        else:
            logger.error(f"Unsupported file type: {file_info['extension']}")
            return jsonify({'error': f'Unsupported file type: {file_info["extension"]}'}), 400
 
        db_manager.save_document(
            session['user_id'],
            session['session_id'],
            filename,
            file_info['mime_type'],
            file_info['size']
        )
 
        os.remove(file_path)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[upload_file] Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to process file'}), 500
 
@app.route('/upload_github', methods=['POST'])
@rate_limit
def upload_github():
    """Process GitHub repository"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        repo_url = data.get('repo_url', '').strip()
        
        if not repo_url:
            return jsonify({'error': 'Repository URL is required'}), 400
        
        # Process GitHub repository
        logger.info(f"Processing GitHub repository: {repo_url}")
        processed_chunks = codebase_processor.process_codebase(repo_url, 'github')
        vector_store.add_codebase(f"code_{session['session_id']}", processed_chunks, repo_url)
        logger.info(f"Added {len(processed_chunks)} GitHub codebase chunks")
        
        # Save to database
        db_manager.save_document(
            session['user_id'],
            session['session_id'],
            repo_url,
            'application/x-git',
            0  # Size not applicable for GitHub repos
        )
        
        return jsonify({
            'type': 'codebase',
            'source': repo_url,
            'message': f'Repository "{repo_url}" processed successfully!',
            'chunks': len(processed_chunks),
            'files_processed': len([c for c in processed_chunks if c.get('chunk_type') == 'code'])
        })
        
    except Exception as e:
        logger.error(f"[upload_github] Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to process repository: {str(e)}'}), 500

@app.route('/get_chat_sessions')
def get_chat_sessions():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        sessions = db_manager.get_user_sessions(session['user_id'])
        return jsonify(sessions)
    except Exception as e:
        logger.error(f"[get_chat_sessions] Error: {e}")
        return jsonify({'error': 'Failed to get chat sessions'}), 500
 
@app.route('/load_session/<session_id>')
def load_session(session_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        messages = db_manager.get_session_messages(session['user_id'], session_id)
        return jsonify(messages)
    except Exception as e:
        logger.error(f"[load_session] Error: {e}")
        return jsonify({'error': 'Failed to load session'}), 500
 
@app.route('/new_session')
def new_session():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    session['session_id'] = f"sess_{str(uuid.uuid4()).replace('-', '')}"
    return jsonify({'session_id': session['session_id']})
 
@app.route('/logout')
def logout():
    if 'user_id' in session and 'session_id' in session:
        try:
            vector_store.delete_collection(session['session_id'])
            vector_store.delete_collection(f"code_{session['session_id']}")
        except Exception as e:
            logger.error(f"[logout] Cleanup error: {e}")
        session.clear()
        flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))
 
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404
 
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
 
@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = "ok"
        try:
            db_manager.get_connection().close()
        except:
            db_status = "error"
        
        # Test vector store connection
        vector_status = "ok"
        try:
            vector_store.connect_to_milvus()
        except:
            vector_status = "error"
        
        # Test LLM connections
        llm_status = llm_client.test_connections()
        
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'vector_store': vector_status,
            'llm_local': llm_status.get('local', False),
            'llm_openai': llm_status.get('openai', False)
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    try:
        db_manager.init_database()
        logger.info("Database initialized successfully!")
        
        # Test LLM connections on startup
        llm_status = llm_client.test_connections()
        logger.info(f"LLM Status - Local: {llm_status['local']}, OpenAI: {llm_status['openai']}")
        
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
    app.run(debug=os.getenv('DEBUG', 'true').lower() == 'true', host='0.0.0.0', port=5000)