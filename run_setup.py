
#!/usr/bin/env python3
"""
Setup script for Enigma Super AI Bot
This script helps set up the environment and database
"""

import os
import sys
import subprocess
import pymysql
from dotenv import load_dotenv

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        print("📚 Downloading NLTK data...")
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {str(e)}")
        return False

def setup_environment():
    """Setup environment file"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("📝 Creating .env file from template...")
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created. Please update with your configuration.")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    return True

def test_database_connection():
    """Test MySQL database connection"""
    load_dotenv()
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            charset='utf8mb4'
        )
        print("✅ MySQL connection successful")
        
        # Create database if it doesn't exist
        cursor = connection.cursor()
        db_name = os.getenv('DB_NAME', 'enigma_ai_bot')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"✅ Database '{db_name}' ready")
        
        connection.close()
        return True
    except Exception as e:
        print(f"❌ MySQL connection failed: {str(e)}")
        print("Please make sure XAMPP is running and MySQL is started")
        return False

def test_milvus_connection():
    """Test Milvus connection"""
    try:
        from pymilvus import connections
        connections.connect(
            alias="default",
            host=os.getenv('MILVUS_HOST', 'localhost'),
            port=os.getenv('MILVUS_PORT', '19530')
        )
        print("✅ Milvus connection successful")
        return True
    except Exception as e:
        print(f"❌ Milvus connection failed: {str(e)}")
        print("Please make sure Milvus is running in Docker")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'static/uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def main():
    """Main setup function"""
    print("🚀 Setting up Enigma Super AI Bot")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    # Setup environment
    if not setup_environment():
        return
    
    # Create directories
    create_directories()
    
    # Test database connection
    if not test_database_connection():
        print("⚠️  Database connection failed. Please check your MySQL setup.")
    
    # Test Milvus connection
    if not test_milvus_connection():
        print("⚠️  Milvus connection failed. Please check your Docker setup.")
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Update .env file with your configuration")
    print("2. Make sure XAMPP MySQL is running")
    print("3. Make sure Milvus is running in Docker")
    print("4. Run: python app.py")

if __name__ == "__main__":
    main()
