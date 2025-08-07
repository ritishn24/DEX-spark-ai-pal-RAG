
# Enigma Super AI Bot

A modular ChatGPT-style chatbot web application built with Flask backend and HTML frontend.

## Features

- User authentication with MySQL
- Document processing (.docx files)
- Image analysis and description
- Vector database storage with Milvus
- Chat history persistence
- Responsive web interface
- Fallback to OpenAI API if local LLM fails

## Prerequisites

- Python 3.8+
- XAMPP (for MySQL)
- Docker (for Milvus)
- Local LLM server running at http://192.168.229.27:8000

## Installation

1. **Clone/Download the project**
   ```bash
   mkdir Enigma_super_ai_bot
   cd Enigma_super_ai_bot
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

5. **Setup MySQL database**
   - Start XAMPP and ensure MySQL is running
   - Create database and tables (see database_setup.sql)

6. **Setup Milvus**
   ```bash
   docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
   ```

7. **Configure environment**
   - Copy `.env.example` to `.env`
   - Update configuration values

8. **Run the application**
   ```bash
   python app.py
   ```

## Project Structure

```
Enigma_super_ai_bot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── database_setup.sql    # MySQL database schema
├── utils/               # Backend utilities
│   ├── __init__.py
│   ├── auth.py          # Authentication logic
│   ├── database.py      # Database operations
│   ├── document_processor.py  # Document processing
│   ├── image_processor.py     # Image processing
│   ├── vector_store.py        # Milvus operations
│   └── llm_client.py          # LLM integration
├── templates/           # HTML templates
│   ├── login.html
│   └── chat.html
└── static/             # Static files
    ├── css/
    ├── js/
    └── uploads/        # Temporary file uploads
```

## Usage

1. Open browser and navigate to `http://localhost:5000`
2. Register a new account or login
3. Start chatting with the AI
4. Upload documents (.docx) or images for analysis
5. View chat history in the sidebar

## API Endpoints

- `GET /` - Home page (redirects to login)
- `GET /login` - Login page
- `POST /login` - Process login
- `POST /register` - User registration
- `GET /chat` - Chat interface
- `POST /chat` - Send message
- `POST /upload` - File upload
- `GET /logout` - Logout and cleanup

## Configuration

Update `.env` file with your settings:
- Database credentials
- LLM server URL
- OpenAI API key (fallback)
- Upload settings
