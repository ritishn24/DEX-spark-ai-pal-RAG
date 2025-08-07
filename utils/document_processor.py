import os
import nltk
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={
                "device": "cpu",  # or 'cuda' if available
                "trust_remote_code": True
            }
        )

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm else embedding

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text.strip())

            return text
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return []

    def preprocess_text(self, text):
        """Preprocess text using NLTK"""
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words = word_tokenize(text)

            processed_words = []
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    processed_words.append(self.lemmatizer.lemmatize(word))

            return ' '.join(processed_words)
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return text

    def chunk_text(self, text, max_chunk_size=500):
        """Split text into chunks"""
        try:
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks
        except Exception as e:
            print(f"Error chunking text: {str(e)}")
            return [text]

    def process_document(self, file_path):
        """Process document and return chunks with embeddings"""
        try:
            paragraphs = self.extract_text_from_docx(file_path)
            processed_chunks = []

            for paragraph in paragraphs:
                if paragraph.strip():
                    processed_text = self.preprocess_text(paragraph)
                    chunks = self.chunk_text(processed_text)

                    for chunk in chunks:
                        if chunk.strip():
                            embedding = self.normalize_embedding(self.embeddings.embed_query(chunk))
                            processed_chunks.append({
                                'text': chunk,
                                'original_text': paragraph,
                                'embedding': embedding
                            })

            return processed_chunks
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return []

    def get_embedding(self, text):
        """Get embedding for text"""
        try:
            return self.normalize_embedding(self.embeddings.embed_query(text))
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None
