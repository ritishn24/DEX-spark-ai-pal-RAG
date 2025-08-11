import os
import git
import zipfile
import tempfile
import shutil
import requests
from pathlib import Path
from typing import List, Dict, Any
import mimetypes
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# =================================================================
# Nomic API Client for Embeddings
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
            
            if response.status_code == 413:
                logger.error(f"Failed to get embedding from Nomic API: 413 Request Entity Too Large. Text content is likely too big.")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get('embeddings')
            
            if not embeddings or not embeddings[0]:
                logger.error("Nomic API returned an empty or invalid embedding.")
                return None
            
            embedding = np.array(embeddings[0])
            norm = np.linalg.norm(embedding)
            return (embedding / norm).tolist() if norm else embedding.tolist()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embedding from Nomic API: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting Nomic embedding: {e}")
            return None

# =================================================================
# Codebase Processing Logic
# =================================================================
class CodebaseProcessor:
    def __init__(self):
        # Initialize Nomic API client directly
        self.api_client = None
        try:
            self.api_client = NomicAPIClient()
            logger.info("✅ Successfully loaded Nomic API client for codebase processing")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Nomic API client: {str(e)}")
            raise Exception("Could not initialize Nomic API client.")
        
        self.code_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.jsx': 'javascript',
            '.tsx': 'typescript', '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.cs': 'csharp', '.php': 'php', '.rb': 'ruby', '.go': 'go', '.rs': 'rust',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala', '.r': 'r', '.m': 'matlab',
            '.sql': 'sql', '.html': 'html', '.css': 'css', '.scss': 'scss', '.less': 'less',
            '.json': 'json', '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml', '.md': 'markdown',
            '.txt': 'text'
        }
        
        self.ignore_patterns = {
            '__pycache__', '.git', '.gitignore', 'node_modules', '.env', '.DS_Store',
            'Thumbs.db', '.vscode', '.idea', '*.pyc', '*.pyo', '*.pyd', '.pytest_cache',
            'coverage', '.coverage', 'dist', 'build', '*.egg-info', '.tox', 'venv', 'env'
        }
    
    def get_embedding(self, text: str, task_type: str = 'search_document') -> List[float] | None:
        """
        Public method to get an embedding for a given text.
        This restores the original method signature for other parts of the application.
        """
        return self.api_client.get_embedding(text, task_type=task_type)

    def clone_github_repo(self, repo_url: str, temp_dir: str) -> str:
        """Clone GitHub repository to temporary directory"""
        try:
            if repo_url.startswith('https://github.com/'):
                clone_url = repo_url
            elif repo_url.startswith('github.com/'):
                clone_url = f"https://{repo_url}"
            else:
                clone_url = f"https://github.com/{repo_url}"
            
            if not clone_url.endswith('.git'):
                clone_url += '.git'
            
            repo_path = os.path.join(temp_dir, 'repo')
            git.Repo.clone_from(clone_url, repo_path, depth=1)
            return repo_path
            
        except Exception as e:
            logger.error(f"Error cloning repository: {str(e)}")
            raise Exception(f"Failed to clone repository: {str(e)}")

    def extract_zip_file(self, zip_path: str, temp_dir: str) -> str:
        """Extract ZIP file to temporary directory"""
        try:
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            contents = os.listdir(extract_path)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_path, contents[0])):
                return os.path.join(extract_path, contents[0])
            
            return extract_path
            
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {str(e)}")
            raise Exception(f"Failed to extract ZIP file: {str(e)}")

    def should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns"""
        path_parts = Path(file_path).parts
        for part in path_parts:
            if any(pattern in part for pattern in self.ignore_patterns):
                return True
        return False

    def get_file_language(self, file_path: str) -> str:
        """Determine programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.code_extensions.get(ext, 'text')
    
    def analyze_file_structure(self, root_path: str) -> Dict[str, Any]:
        """Analyze codebase structure and generate summary"""
        structure = {
            'total_files': 0, 'languages': {}, 'directories': [], 'large_files': [], 'file_tree': {}
        }
        
        try:
            for root, dirs, files in os.walk(root_path):
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.ignore_patterns)]
                rel_root = os.path.relpath(root, root_path)
                if rel_root != '.':
                    structure['directories'].append(rel_root)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_path)
                    
                    if self.should_ignore_file(rel_path):
                        continue
                    
                    structure['total_files'] += 1
                    language = self.get_file_language(file_path)
                    structure['languages'][language] = structure['languages'].get(language, 0) + 1
                    
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > 100000:
                            structure['large_files'].append({
                                'path': rel_path, 'size': file_size, 'language': language
                            })
                    except:
                        pass
            return structure
        except Exception as e:
            logger.error(f"Error analyzing file structure: {str(e)}")
            return structure

    def _read_file_and_chunk_stream(self, file_path: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Reads a file line-by-line and chunks it into a list of dictionaries.
        This approach is memory-efficient and ideal for very large files.
        """
        chunks = []
        current_chunk_lines = []
        current_chunk_size = 0
        chunk_index = 0

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_size = len(line)
                    if current_chunk_size + line_size > max_chunk_size and current_chunk_lines:
                        chunk_content = "".join(current_chunk_lines)
                        chunks.append({
                            'content': chunk_content,
                            'file_path': file_path,
                            'chunk_index': chunk_index
                        })
                        current_chunk_lines = [line]
                        current_chunk_size = line_size
                        chunk_index += 1
                    else:
                        current_chunk_lines.append(line)
                        current_chunk_size += line_size
            
            if current_chunk_lines:
                chunk_content = "".join(current_chunk_lines)
                chunks.append({
                    'content': chunk_content,
                    'file_path': file_path,
                    'chunk_index': chunk_index
                })

        except Exception as e:
            logger.error(f"Error processing file {file_path} line-by-line: {e}")
            return []
        
        return chunks

    def process_codebase(self, source_path: str, source_type: str = 'directory') -> List[Dict[str, Any]]:
        """Process codebase and return chunks with embeddings"""
        processed_chunks = []
        temp_dir = None
        
        if not self.api_client:
            raise Exception("Nomic API client is not initialized.")
        
        try:
            if source_type == 'github':
                temp_dir = tempfile.mkdtemp()
                root_path = self.clone_github_repo(source_path, temp_dir)
            elif source_type == 'zip':
                temp_dir = tempfile.mkdtemp()
                root_path = self.extract_zip_file(source_path, temp_dir)
            else:
                root_path = source_path
            
            structure = self.analyze_file_structure(root_path)
            structure_summary = self.generate_structure_summary(structure)
            
            structure_embedding = self.api_client.get_embedding(structure_summary, task_type='search_document')
            if structure_embedding is not None and len(structure_embedding) > 0:
                processed_chunks.append({
                    'content': structure_summary, 'file_path': 'PROJECT_STRUCTURE.md',
                    'chunk_type': 'structure', 'language': 'markdown',
                    'embedding': structure_embedding, 'metadata': structure
                })
            
            for root, dirs, files in os.walk(root_path):
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.ignore_patterns)]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_path)
                    
                    if self.should_ignore_file(rel_path) or not self.is_text_file(file_path):
                        continue
                    
                    file_chunks = self._read_file_and_chunk_stream(file_path)
                    
                    if not file_chunks:
                        continue
                    
                    batch_texts = []
                    for chunk in file_chunks:
                        chunk_text = f"File: {rel_path}\nLanguage: {self.get_file_language(rel_path)}\n\n{chunk['content']}"
                        batch_texts.append(chunk_text)
                    
                    if batch_texts:
                        embeddings = [self.api_client.get_embedding(text, task_type='search_document') for text in batch_texts]

                        for i, chunk in enumerate(file_chunks):
                            embedding = embeddings[i]
                            if embedding:
                                processed_chunks.append({
                                    'content': chunk['content'], 'file_path': rel_path,
                                    'chunk_type': 'code', 'language': self.get_file_language(rel_path),
                                    'start_line': 0,
                                    'end_line': 0,
                                    'embedding': embedding, 'full_context': batch_texts[i]
                                })
                            else:
                                logger.warning(f"Failed to generate embedding for chunk {i} of {rel_path}")

            logger.info(f"Successfully processed {len(processed_chunks)} code chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing codebase: {str(e)}")
            raise
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext in self.code_extensions:
                return True
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('text/'):
                return True
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return False
            return True
        except Exception:
            return False

    def generate_structure_summary(self, structure: Dict[str, Any]) -> str:
        """Generate a summary of the codebase structure"""
        summary = f"""# Codebase Structure Summary

## Overview
- Total Files: {structure['total_files']}
- Directories: {len(structure['directories'])}

## Languages Distribution
"""
        for lang, count in sorted(structure['languages'].items(), key=lambda x: x[1], reverse=True):
            summary += f"- {lang.title()}: {count} files\n"
        
        if structure['directories']:
            summary += "\n## Directory Structure\n"
            for directory in sorted(structure['directories'])[:20]:
                summary += f"- {directory}\n"
        
        if structure['large_files']:
            summary += "\n## Large Files (>100KB)\n"
            for file_info in structure['large_files'][:10]:
                size_kb = file_info['size'] // 1024
                summary += f"- {file_info['path']} ({size_kb}KB, {file_info['language']})\n"
        
        return summary
