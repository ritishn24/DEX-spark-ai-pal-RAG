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
class LLMClient:
    def __init__(self):
        self.llm_server_url = os.getenv('LLM_SERVER_URL', 'http://localhost:8000/v1/chat/completions')
        self.llm_model_path = os.getenv('LLM_MODEL_PATH', '/root/.cache/huggingface/')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
 
    def generate_response(self, user_message, context="", image_url=None, max_tokens=10000, temperature=0.1):
        try:
            local_response = self._call_local_llm(user_message, context, image_url, max_tokens, temperature)
            if local_response:
                return local_response
        except Exception as e:
            print(f"[Local LLM Error] {e}")
 
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client is not initialized")
            return self._call_openai_gpt4o(user_message, context, image_url, max_tokens, temperature)
        except Exception as e:
            print(f"[OpenAI GPT-4o Error] {e}")
 
        return "I'm sorry, I'm currently unable to process your request. Please try again later."
 
    def _call_local_llm(self, user_message, context="", image_url=None, max_tokens=10000, temperature=0.1):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer only based on the document chunks provided in the context. "
            "If the answer is not clearly present in the context, respond with: "
            "\"This information is not found in the uploaded document.\" "
            "Do not use any external knowledge."
        )
 
        if image_url:
            message_content = [
                {"type": "text", "text": self._format_message_with_context(user_message, context)},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        else:
            message_content = self._format_message_with_context(user_message, context)
 
        payload = {
            "model": self.llm_model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
 
        response = requests.post(self.llm_server_url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"[Local LLM] HTTP {response.status_code} - {response.text}")
        return None
 
    def _call_openai_gpt4o(self, user_message, context="", image_url=None, max_tokens=1000, temperature=0.1):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer only based on the document chunks provided in the context. "
            "If the answer is not clearly present in the context, respond with: "
            "\"This information is not found in the uploaded document.\" "
            "Do not use any external knowledge."
        )
 
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
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
 
    def _format_message_with_context(self, user_message, context=""):
        return f"Context:\n{context}\n\nUser Question: {user_message}" if context.strip() else user_message
 
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
 
        db_manager.save_message(session['user_id'], session['session_id'], 'user', user_message)
 
        history = db_manager.get_session_messages(session['user_id'], session['session_id'])
        memory_context = "\n".join([f"{m['role'].capitalize()}: {m['message']}" for m in history[-10:]])
 
        vector_context = ""
        
        # Different context retrieval based on chat mode
        if chat_mode == 'codebase':
            if vector_store.collection_exists(f"code_{session['session_id']}"):
                relevant_docs = vector_store.search_codebase(f"code_{session['session_id']}", user_message)
                vector_context = self._format_code_context(relevant_docs)
        else:
            if vector_store.collection_exists(session['session_id']):
                relevant_docs = vector_store.search_documents(session['session_id'], user_message)
                vector_context = "\n".join([doc.get('text', '') for doc in relevant_docs])
 
        full_context = f"""Chat History:
{memory_context.strip()}
 
Relevant Docs:
{vector_context.strip()}"""
        
        # Customize system prompt based on mode
        system_context = self._get_system_context(chat_mode)
        ai_response = llm_client.generate_response(user_message, full_context, system_context=system_context)
 
        db_manager.save_message(session['user_id'], session['session_id'], 'assistant', ai_response)
 
        return jsonify({
            'user_message': user_message,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"[send_message] Error: {e}")
        return jsonify({'error': 'Failed to process message'}), 500
 
def _format_code_context(self, relevant_docs):
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

def _get_system_context(self, mode):
    """Get system context based on chat mode"""
    if mode == 'codebase':
        return """You are a code analysis expert. Help users understand codebases, explain functionality, 
        suggest improvements, and answer questions about code structure and implementation. 
        Provide clear explanations with code examples when relevant."""
    elif mode == 'image':
        return """You are an image analysis expert. Describe images in detail, identify objects, 
        people, text, and provide contextual information about what you see."""
    else:
        return """You are a helpful AI assistant. Answer questions based on the provided context 
        and help users with their queries."""

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
 
        filename = secure_filename(file.filename or "uploaded_file")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Validate file
        is_valid, error_msg, file_info = file_validator.validate_file(file_path, filename)
        if not is_valid:
            os.remove(file_path)
            return jsonify({'error': error_msg}), 400
 
        result = {}
 
        if upload_type == 'codebase' and filename.lower().endswith('.zip'):
            # Process codebase from ZIP
            processed_chunks = codebase_processor.process_codebase(file_path, 'zip')
            vector_store.add_codebase(f"code_{session['session_id']}", processed_chunks, filename)
            result = {
                'type': 'codebase',
                'filename': filename,
                'message': f'Codebase "{filename}" processed successfully!',
                'chunks': len(processed_chunks),
                'files_processed': len([c for c in processed_chunks if c.get('chunk_type') == 'code'])
            }
        elif file_info['extension'] in ['docx', 'pdf', 'txt', 'md']:
            # Process document
            if file_info['extension'] == 'docx':
                processed_text = doc_processor.process_document(file_path)
            else:
                # Handle other document types
                processed_text = self._process_other_documents(file_path, file_info['extension'])
            
            vector_store.add_documents(session['session_id'], processed_text, filename)
            result = {
                'type': 'document',
                'filename': filename,
                'message': f'Document "{filename}" processed successfully!',
                'chunks': len(processed_text)
            }
        elif file_info['extension'] in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
            # Process image
            ai_response = img_processor.analyze_with_ai(file_path)
            result = {
                'type': 'image',
                'filename': filename,
                'description': ai_response,
                'message': f'Image "{filename}" analyzed successfully!'
            }
        else:
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
        print(f"[upload_file] Error: {e}")
        return jsonify({'error': 'Failed to process file'}), 500
 
def _process_other_documents(self, file_path, extension):
    """Process non-DOCX documents"""
    try:
        if extension in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking for text files
            chunks = []
            sentences = content.split('\n\n')  # Split by paragraphs
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    embedding = doc_processor.get_embedding(sentence)
                    chunks.append({
                        'text': sentence.strip(),
                        'original_text': sentence.strip(),
                        'embedding': embedding
                    })
            
            return chunks
        elif extension == 'pdf':
            # PDF processing would require additional library like PyPDF2
            return [{'text': 'PDF processing not implemented yet', 'original_text': 'PDF processing not implemented yet', 'embedding': [0]*768}]
        
        return []
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return []

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
        processed_chunks = codebase_processor.process_codebase(repo_url, 'github')
        vector_store.add_codebase(f"code_{session['session_id']}", processed_chunks, repo_url)
        
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
        print(f"[upload_github] Error: {e}")
        return jsonify({'error': f'Failed to process repository: {str(e)}'}), 500

@app.route('/get_chat_sessions')
def get_chat_sessions():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        sessions = db_manager.get_user_sessions(session['user_id'])
        return jsonify(sessions)
    except Exception as e:
        print(f"[get_chat_sessions] Error: {e}")
        return jsonify({'error': 'Failed to get chat sessions'}), 500
 
@app.route('/load_session/<session_id>')
def load_session(session_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        messages = db_manager.get_session_messages(session['user_id'], session_id)
        return jsonify(messages)
    except Exception as e:
        print(f"[load_session] Error: {e}")
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
            print(f"[logout] Cleanup error: {e}")
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

if __name__ == '__main__':
    try:
        db_manager.init_database()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
    app.run(debug=os.getenv('DEBUG', 'true').lower() == 'true', host='0.0.0.0', port=5000)
   
 