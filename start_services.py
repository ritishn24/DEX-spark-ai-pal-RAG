
#!/usr/bin/env python3
"""
Service starter for Enigma Super AI Bot
This script helps start required services
"""

import os
import sys
import subprocess
import time
import requests

def start_milvus_docker():
    """Start Milvus in Docker"""
    try:
        print("🐳 Starting Milvus in Docker...")
        
        # Check if container already exists
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=milvus'], 
                              capture_output=True, text=True)
        
        if 'milvus' in result.stdout:
            # Container exists, start it
            subprocess.run(['docker', 'start', 'milvus'], check=True)
            print("✅ Milvus container started")
        else:
            # Create and start new container
            subprocess.run([
                'docker', 'run', '-d',
                '--name', 'milvus',
                '-p', '19530:19530',
                '-p', '9091:9091',
                'milvusdb/milvus:latest'
            ], check=True)
            print("✅ Milvus container created and started")
        
        # Wait for Milvus to be ready
        print("⏳ Waiting for Milvus to be ready...")
        time.sleep(10)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Milvus: {str(e)}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        return False

def check_mysql_service():
    """Check if MySQL is running"""
    try:
        import pymysql
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            charset='utf8mb4'
        )
        connection.close()
        print("✅ MySQL is running")
        return True
    except Exception:
        print("❌ MySQL is not running. Please start XAMPP MySQL.")
        return False

def check_llm_server():
    """Check if LLM server is running"""
    try:
        response = requests.get('http://192.168.229.27:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ LLM server is running")
            return True
        else:
            print("⚠️  LLM server responded but with error")
            return False
    except requests.exceptions.RequestException:
        print("⚠️  LLM server is not responding. Fallback to OpenAI will be used.")
        return False

def main():
    """Main service starter"""
    print("🚀 Starting services for Enigma Super AI Bot")
    print("=" * 40)
    
    # Start Milvus
    if not start_milvus_docker():
        print("⚠️  Milvus failed to start. Vector search may not work.")
    
    # Check MySQL
    if not check_mysql_service():
        print("⚠️  Please start XAMPP and ensure MySQL is running.")
    
    # Check LLM server
    if not check_llm_server():
        print("⚠️  Local LLM server not available. Make sure to set OPENAI_API_KEY.")
    
    print("\n" + "=" * 40)
    print("✅ Service check completed!")
    print("\nTo start the application, run: python app.py")

if __name__ == "__main__":
    main()
