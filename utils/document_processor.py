import os
import nltk
from docx import Document
import re
import numpy as np
import PyPDF2
import logging
import requests
from typing import List, Dict, Any

# =================================================================
# Nomic API Client for Embeddings (Integrated)
# =================================================================
class NomicAPIClient:
    def __init__(self, dimensionality: int = 768):
        self.api_key = os.getenv('NOMIC_API_KEY')
        self.api_url = os.getenv('NOMIC_API_URL', 'https://api-atlas.nomic.ai/v1/embedding/text')
        self.dimensionality = dimensionality
        self.model = 'nomic-embed-text-v1.5'

        if not self.api_key:
            logger.error("NOMIC_API_KEY environment variable is not set. Cannot use Nomic API.")
            raise ValueError("NOMIC_API_KEY is not set.")
        
        logger.info(f"Nomic API Client initialized. Using URL: {self.api_url}, Dimensionality: {self.dimensionality}")

    def get_embedding(self, text: str, task_type: str = 'search_document') -> List[float] | None:
        """
        Get embedding for a single text string using Nomic's API.
        
        Args:
            text: The text to embed.
            task_type: The task type for the embedding (e.g., 'search_document', 'search_query').
        
        Returns:
            A list representing the embedding vector, or None if the request fails.
        """
        try:
            payload = {
                "model": self.model,
                "texts": [text],
                "task_type": task_type,
                "dimensionality": self.dimensionality
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get('embeddings')
            
            if not embeddings or not embeddings[0]:
                logger.error("Nomic API returned an empty or invalid embedding.")
                return None
            
            # Normalize the embedding vector
            embedding = np.array(embeddings[0])
            norm = np.linalg.norm(embedding)
            return (embedding / norm).tolist() if norm else embedding.tolist()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embedding from Nomic API: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting Nomic embedding: {e}")
            return None

# Configure logging
logger = logging.getLogger(__name__)

# =================================================================
# Document Processing Logic
# =================================================================
class DocumentProcessor:
    def __init__(self):
        # We assume NLTK data is downloaded in the container
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Initialize Nomic API client
        self.api_client = None
        try:
            self.api_client = NomicAPIClient()
            logger.info("✅ Successfully loaded Nomic API client for document processing")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Nomic API client: {str(e)}")
            raise Exception("Could not initialize Nomic API client.")

    def get_embedding(self, text: str, task_type: str = 'search_document') -> List[float] | None:
        """
        Public method to get an embedding for a given text.
        This restores the original method signature for other parts of the application.
        """
        return self.api_client.get_embedding(text, task_type=task_type)

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF file"""
        try:
            logger.info(f"Extracting text from PDF: {file_path}")
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(page_text.strip())
                            logger.info(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            logger.info(f"Total pages processed: {len(text_content)}")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return []

    def extract_text_from_docx(self, file_path: str) -> List[str]:
        """Extract text from DOCX file"""
        try:
            logger.info(f"Extracting text from DOCX: {file_path}")
            doc = Document(file_path)
            text = []

            for paragraph in doc.paragraphs:
                if paragraph.text and paragraph.text.strip():
                    text.append(paragraph.text.strip())

            logger.info(f"Extracted {len(text)} paragraphs from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return []
    
    def extract_text_from_txt(self, file_path: str) -> List[str]:
        """Extract text from TXT/MD files"""
        try:
            logger.info(f"Extracting text from TXT/MD: {file_path}")
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    if content and content.strip():
                        # Split by paragraphs for better chunking
                        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                        logger.info(f"Extracted {len(paragraphs)} paragraphs from text file")
                        return paragraphs
                    break
                except UnicodeDecodeError:
                    continue
            
            return []
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            return []

    def chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        try:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return [text]

    def process_document(self, file_path: str, file_extension: str | None = None) -> List[Dict[str, Any]]:
        """Process document and return chunks with embeddings"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            if not file_extension:
                file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            if file_extension == 'pdf':
                paragraphs = self.extract_text_from_pdf(file_path)
            elif file_extension in ['docx', 'doc']:
                paragraphs = self.extract_text_from_docx(file_path)
            elif file_extension in ['txt', 'md']:
                paragraphs = self.extract_text_from_txt(file_path)
            else:
                logger.error(f"Unsupported file extension: {file_extension}")
                return []
            
            if not paragraphs:
                logger.error("No text extracted from document")
                return []
            
            logger.info(f"Extracted {len(paragraphs)} text sections")
            processed_chunks = []

            for paragraph in paragraphs:
                if paragraph.strip():
                    chunks = self.chunk_text(paragraph)
                    for chunk in chunks:
                        if chunk.strip():
                            try:
                                # Use Nomic API Client for embedding
                                embedding = self.api_client.get_embedding(chunk, task_type='search_document')
                                if embedding:
                                    processed_chunks.append({
                                        'text': chunk,
                                        'original_text': paragraph,
                                        'embedding': embedding
                                    })
                                    logger.debug(f"Created chunk with {len(embedding)} dimensions")
                            except Exception as e:
                                logger.error(f"Error creating embedding for chunk: {str(e)}")
                                continue

            logger.info(f"Successfully processed {len(processed_chunks)} chunks")
            return processed_chunks
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return []
