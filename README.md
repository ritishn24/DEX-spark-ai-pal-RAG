
# AI-Powered Multi-Modal Chat Application

A comprehensive AI-powered chat application with multi-modal capabilities including document analysis, codebase exploration, and image understanding.

## Features

### 🤖 Multi-Modal Chat Interface
- **General Chat**: RAG-enabled conversations with document context
- **Codebase Analysis**: Interactive code exploration and Q&A
- **Image Analysis**: Detailed image description and contextual understanding

### 📚 Document Processing & RAG
- Support for multiple formats: PDF, DOCX, TXT, Markdown
- Intelligent document chunking and embedding generation
- Semantic search with Milvus vector database
- Context-aware responses based on uploaded documents

### 💻 Codebase Exploration
- **GitHub Integration**: Direct repository cloning and analysis
- **ZIP Upload**: Process local codebases via file upload
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, and more
- **Intelligent Code Analysis**: Structure understanding, documentation generation, debugging assistance

### 🖼️ Image Understanding
- Advanced image analysis with AI vision models
- Support for JPG, PNG, GIF, WebP, BMP formats
- Contextual image descriptions and Q&A

### 🔐 Security & Performance
- User authentication and session management
- Rate limiting and file validation
- Secure file handling with type verification
- Responsive design for all devices
- Comprehensive error handling and logging

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- MySQL (or use Docker setup)
- Docker (for Milvus)
- Optional: Local LLM server or OpenAI API key

## Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-chat-application
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Open http://localhost:5000
   - Register a new account or use default admin credentials

## Manual Installation

1. **Setup the project**
   ```bash
   git clone <repository-url>
   cd ai-chat-application
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```

5. **Setup databases**
   ```bash
   # Start MySQL (via XAMPP or standalone)
   # Import database_setup.sql
   
   # Start Milvus
   docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.0
   ```

6. **Configure environment**
   ```bash
   cp .env.example .env
   # Update .env with your settings
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

## Architecture Overview

```
CHATTY-SPARK-AI-PAL RAG/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-service setup
├── .env.example          # Environment configuration template
├── database_setup.sql    # MySQL database schema
├── utils/               # Backend utilities
│   ├── __init__.py
│   ├── auth.py          # Authentication logic
│   ├── database.py      # Database operations
│   ├── document_processor.py  # Document processing
│   ├── image_processor.py     # Image processing
│   ├── codebase_processor.py  # Code analysis
│   ├── vector_store.py        # Milvus operations
│   ├── llm_client.py          # LLM integration
│   ├── rate_limiter.py        # Rate limiting
│   └── file_validator.py      # File validation
├── templates/           # HTML templates
│   ├── login.html
│   └── chat.html
├── static/             # Static files
│   └── uploads/        # Temporary file uploads
└── logs/               # Application logs
```

## Feature Usage Guide

### 💬 General Chat Mode
1. Select "General Chat" mode
2. Upload documents (PDF, DOCX, TXT, MD)
3. Ask questions about the uploaded content
4. Enjoy context-aware responses

### 💻 Codebase Analysis Mode
1. Switch to "Codebase Analysis" mode
2. **Option A**: Enter GitHub repository URL (e.g., "username/repo")
3. **Option B**: Upload ZIP file containing your code
4. Ask questions about:
   - Code structure and architecture
   - Function explanations
   - Best practices and improvements
   - Debugging assistance

### 🖼️ Image Analysis Mode
1. Select "Image Analysis" mode
2. Upload an image (JPG, PNG, GIF, WebP, BMP)
3. Ask questions about the image content
4. Get detailed descriptions and contextual information


## API Endpoints

### Authentication
- `GET /` - Home page (redirects appropriately)
- `GET /login` - Login/registration page
- `POST /login` - Process login/registration
- `GET /logout` - Logout and cleanup

### Chat Operations
- `GET /chat` - Main chat interface
- `POST /send_message` - Send chat message
- `POST /upload_file` - Upload and process files
- `POST /upload_github` - Process GitHub repository

### Session Management
- `GET /get_chat_sessions` - Get user's chat sessions
- `GET /load_session/<session_id>` - Load specific session
- `GET /new_session` - Create new chat session

## Environment Configuration

```env
# Database Configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=enigma_ai_bot

# Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# AI Models
LLM_SERVER_URL=http://localhost:8000/v1/chat/completions
OPENAI_API_KEY=your_openai_key

# GitHub Integration
GITHUB_TOKEN=your_github_token

# Security
SECRET_KEY=your-secret-key
RATE_LIMIT_PER_MINUTE=60
MAX_FILE_SIZE_MB=50
```

## Supported File Types

### Documents
- **PDF**: Portable Document Format
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **MD**: Markdown files

### Images
- **JPG/JPEG**: JPEG images
- **PNG**: Portable Network Graphics
- **GIF**: Graphics Interchange Format
- **WebP**: Modern web image format
- **BMP**: Bitmap images

### Code Archives
- **ZIP**: Compressed archives
- **TAR**: Tape archives
- **GZ**: Gzip compressed files

## Performance Optimization

### Vector Database
- Milvus provides efficient similarity search
- Automatic indexing for fast retrieval
- Configurable collection management

### Rate Limiting
- 60 requests per minute per user
- 1000 requests per hour per user
- Configurable limits via environment variables

### File Processing
- Intelligent chunking for large documents
- Parallel processing for code analysis
- Efficient embedding generation

## Security Features

### File Validation
- MIME type verification
- File size limits
- Extension whitelist/blacklist
- Malicious file detection

### Authentication
- Secure password hashing
- Session management
- Rate limiting protection
- Input sanitization

## Monitoring & Logging

### Application Logs
- Request/response logging
- Error tracking
- Performance metrics
- Security events

### Health Checks
- Database connectivity
- Vector store status
- LLM availability
- System resources

## Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   ```bash
   docker restart milvus
   # Check if port 19530 is available
   ```

2. **File Upload Errors**
   - Check file size limits
   - Verify file type support
   - Ensure upload directory permissions

3. **GitHub Repository Access**
   - Verify repository URL format
   - Check GitHub token permissions
   - Ensure repository is public or accessible

4. **LLM Response Issues**
   - Check local LLM server status
   - Verify OpenAI API key
   - Review rate limiting settings

### Debug Mode
```bash
export DEBUG=true
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation
