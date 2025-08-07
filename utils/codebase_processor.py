import os
import git
import zipfile
import tempfile
import shutil
import requests
from pathlib import Path
from typing import List, Dict, Any
import mimetypes
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

class CodebaseProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True
            }
        )
        
        # Supported file extensions for code analysis
        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        # Files to ignore during processing
        self.ignore_patterns = {
            '__pycache__', '.git', '.gitignore', 'node_modules', '.env',
            '.DS_Store', 'Thumbs.db', '.vscode', '.idea', '*.pyc',
            '*.pyo', '*.pyd', '.pytest_cache', 'coverage', '.coverage',
            'dist', 'build', '*.egg-info', '.tox', 'venv', 'env'
        }

    def normalize_embedding(self, embedding):
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm else embedding

    def clone_github_repo(self, repo_url: str, temp_dir: str) -> str:
        """Clone GitHub repository to temporary directory"""
        try:
            # Handle different GitHub URL formats
            if repo_url.startswith('https://github.com/'):
                clone_url = repo_url
            elif repo_url.startswith('github.com/'):
                clone_url = f"https://{repo_url}"
            else:
                # Assume it's in format "username/repo"
                clone_url = f"https://github.com/{repo_url}"
            
            if not clone_url.endswith('.git'):
                clone_url += '.git'
            
            repo_path = os.path.join(temp_dir, 'repo')
            git.Repo.clone_from(clone_url, repo_path, depth=1)
            return repo_path
            
        except Exception as e:
            print(f"Error cloning repository: {str(e)}")
            raise Exception(f"Failed to clone repository: {str(e)}")

    def extract_zip_file(self, zip_path: str, temp_dir: str) -> str:
        """Extract ZIP file to temporary directory"""
        try:
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find the root directory (handle nested structures)
            contents = os.listdir(extract_path)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_path, contents[0])):
                return os.path.join(extract_path, contents[0])
            
            return extract_path
            
        except Exception as e:
            print(f"Error extracting ZIP file: {str(e)}")
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

    def read_file_content(self, file_path: str) -> str:
        """Read file content with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                return ""
        
        return ""

    def analyze_file_structure(self, root_path: str) -> Dict[str, Any]:
        """Analyze codebase structure and generate summary"""
        structure = {
            'total_files': 0,
            'languages': {},
            'directories': [],
            'large_files': [],
            'file_tree': {}
        }
        
        try:
            for root, dirs, files in os.walk(root_path):
                # Skip ignored directories
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
                    
                    # Language detection
                    language = self.get_file_language(file_path)
                    structure['languages'][language] = structure['languages'].get(language, 0) + 1
                    
                    # Check file size
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > 100000:  # Files larger than 100KB
                            structure['large_files'].append({
                                'path': rel_path,
                                'size': file_size,
                                'language': language
                            })
                    except:
                        pass
            
            return structure
            
        except Exception as e:
            print(f"Error analyzing file structure: {str(e)}")
            return structure

    def chunk_code_content(self, content: str, file_path: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Split code content into meaningful chunks"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'file_path': file_path,
                    'start_line': i - len(current_chunk) + 1,
                    'end_line': i,
                    'language': self.get_file_language(file_path)
                })
                
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'file_path': file_path,
                'start_line': len(lines) - len(current_chunk) + 1,
                'end_line': len(lines),
                'language': self.get_file_language(file_path)
            })
        
        return chunks

    def process_codebase(self, source_path: str, source_type: str = 'directory') -> List[Dict[str, Any]]:
        """Process codebase and return chunks with embeddings"""
        processed_chunks = []
        temp_dir = None
        
        try:
            if source_type == 'github':
                temp_dir = tempfile.mkdtemp()
                root_path = self.clone_github_repo(source_path, temp_dir)
            elif source_type == 'zip':
                temp_dir = tempfile.mkdtemp()
                root_path = self.extract_zip_file(source_path, temp_dir)
            else:
                root_path = source_path
            
            # Analyze structure first
            structure = self.analyze_file_structure(root_path)
            
            # Add structure summary as first chunk
            structure_summary = self.generate_structure_summary(structure)
            structure_embedding = self.normalize_embedding(
                self.embeddings.embed_query(structure_summary)
            )
            
            processed_chunks.append({
                'content': structure_summary,
                'file_path': 'PROJECT_STRUCTURE.md',
                'chunk_type': 'structure',
                'language': 'markdown',
                'embedding': structure_embedding,
                'metadata': structure
            })
            
            # Process individual files
            for root, dirs, files in os.walk(root_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.ignore_patterns)]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_path)
                    
                    if self.should_ignore_file(rel_path):
                        continue
                    
                    # Skip binary files
                    if not self.is_text_file(file_path):
                        continue
                    
                    content = self.read_file_content(file_path)
                    if not content.strip():
                        continue
                    
                    # Create chunks for the file
                    file_chunks = self.chunk_code_content(content, rel_path)
                    
                    for chunk in file_chunks:
                        # Generate embedding for chunk
                        chunk_text = f"File: {chunk['file_path']}\nLanguage: {chunk['language']}\nLines {chunk['start_line']}-{chunk['end_line']}:\n\n{chunk['content']}"
                        embedding = self.normalize_embedding(
                            self.embeddings.embed_query(chunk_text)
                        )
                        
                        processed_chunks.append({
                            'content': chunk['content'],
                            'file_path': chunk['file_path'],
                            'chunk_type': 'code',
                            'language': chunk['language'],
                            'start_line': chunk['start_line'],
                            'end_line': chunk['end_line'],
                            'embedding': embedding,
                            'full_context': chunk_text
                        })
            
            return processed_chunks
            
        except Exception as e:
            print(f"Error processing codebase: {str(e)}")
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file"""
        try:
            # Check by extension first
            ext = Path(file_path).suffix.lower()
            if ext in self.code_extensions:
                return True
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('text/'):
                return True
            
            # Try to read first few bytes
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:  # Binary file indicator
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
            for directory in sorted(structure['directories'])[:20]:  # Limit to first 20
                summary += f"- {directory}\n"
        
        if structure['large_files']:
            summary += "\n## Large Files (>100KB)\n"
            for file_info in structure['large_files'][:10]:  # Limit to first 10
                size_kb = file_info['size'] // 1024
                summary += f"- {file_info['path']} ({size_kb}KB, {file_info['language']})\n"
        
        return summary

    def get_embedding(self, text: str):
        """Get embedding for text"""
        try:
            return self.normalize_embedding(self.embeddings.embed_query(text))
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None