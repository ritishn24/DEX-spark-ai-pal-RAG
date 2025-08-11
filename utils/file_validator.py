import os
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available, MIME type detection disabled")

from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FileValidator:
    def __init__(self):
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE_MB', 50)) * 1024 * 1024  # Convert to bytes
        
        # Allowed file types and their MIME types
        self.allowed_types = {
            # Documents
            'pdf': ['application/pdf'],
            'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            'doc': ['application/msword'],
            'txt': ['text/plain'],
            'md': ['text/markdown', 'text/x-markdown'],
            
            # Images
            'jpg': ['image/jpeg'],
            'jpeg': ['image/jpeg'],
            'png': ['image/png'],
            'gif': ['image/gif'],
            'webp': ['image/webp'],
            'bmp': ['image/bmp'],
            
            # Archives
            'zip': ['application/zip', 'application/x-zip-compressed'],
            'tar': ['application/x-tar'],
            'gz': ['application/gzip'],
            
            # Code files
            'py': ['text/x-python', 'text/plain'],
            'js': ['application/javascript', 'text/javascript', 'text/plain'],
            'ts': ['application/typescript', 'text/plain'],
            'html': ['text/html'],
            'css': ['text/css'],
            'json': ['application/json', 'text/plain'],
            'xml': ['application/xml', 'text/xml'],
            'yaml': ['application/x-yaml', 'text/yaml', 'text/plain'],
            'yml': ['application/x-yaml', 'text/yaml', 'text/plain'],
        }
        
        # Dangerous file extensions to block
        self.blocked_extensions = {
            'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar',
            'app', 'deb', 'pkg', 'rpm', 'dmg', 'iso', 'msi', 'dll', 'so'
        }

    def validate_file(self, file_path: str, original_filename: str) -> Tuple[bool, Optional[str], Optional[dict]]:
        """
        Validate uploaded file
        Returns: (is_valid, error_message, file_info)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, "File not found", None
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = Path(original_filename).suffix.lower().lstrip('.')
            
            # Check file size
            if file_size > self.max_file_size:
                max_mb = self.max_file_size // (1024 * 1024)
                return False, f"File too large. Maximum size is {max_mb}MB", None
            
            # Check for blocked extensions
            if file_ext in self.blocked_extensions:
                return False, f"File type '{file_ext}' is not allowed for security reasons", None
            
            # Check if extension is allowed
            if file_ext not in self.allowed_types:
                return False, f"File type '{file_ext}' is not supported", None
            
            # Validate MIME type using python-magic
            mime_type = 'unknown'
            if MAGIC_AVAILABLE:
                try:
                    mime_type = magic.from_file(file_path, mime=True)
                    allowed_mimes = self.allowed_types[file_ext]
                    
                    if mime_type not in allowed_mimes:
                        # Some flexibility for text files
                        if file_ext in ['py', 'js', 'ts', 'txt', 'md', 'yaml', 'yml', 'json', 'xml'] and mime_type.startswith('text/'):
                            pass  # Allow text MIME types for code files
                        else:
                            logger.warning(f"MIME type mismatch for {file_ext}: expected {allowed_mimes}, got {mime_type}")
                            # Don't fail validation, just log warning
                    
                except Exception as e:
                    logger.warning(f"MIME type detection failed: {str(e)}")
                    # Continue without MIME validation if magic fails
            else:
                logger.info("MIME type detection skipped (python-magic not available)")
            
            # Additional security checks
            if not self.is_safe_filename(original_filename):
                return False, "Filename contains unsafe characters", None
            
            file_info = {
                'size': file_size,
                'extension': file_ext,
                'mime_type': mime_type,
                'safe_filename': self.sanitize_filename(original_filename)
            }
            
            return True, None, file_info
            
        except Exception as e:
            return False, f"File validation error: {str(e)}", None

    def is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe"""
        # Block dangerous characters and patterns
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
            'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in reserved_names:
            return False
        
        return True

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove or replace dangerous characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '.-_ ':
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        
        safe_filename = ''.join(safe_chars)
        
        # Ensure filename is not too long
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:255-len(ext)] + ext
        
        return safe_filename

    def get_file_category(self, file_extension: str) -> str:
        """Get file category based on extension"""
        document_types = {'pdf', 'docx', 'doc', 'txt', 'md'}
        image_types = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'}
        archive_types = {'zip', 'tar', 'gz'}
        code_types = {'py', 'js', 'ts', 'html', 'css', 'json', 'xml', 'yaml', 'yml'}
        
        if file_extension in document_types:
            return 'document'
        elif file_extension in image_types:
            return 'image'
        elif file_extension in archive_types:
            return 'archive'
        elif file_extension in code_types:
            return 'code'
        else:
            return 'unknown'