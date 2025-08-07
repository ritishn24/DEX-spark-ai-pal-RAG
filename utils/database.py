
import os
import pymysql
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.database = os.getenv('DB_NAME', 'enigma_ai_bot')
    
    def get_connection(self):
        """Get database connection"""
        try:
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            raise
    
    def init_database(self):
        """Initialize database tables"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chat_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_session (user_id, session_id),
                    INDEX idx_timestamp (timestamp)
                )
            """)
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    file_size INT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            connection.commit()
            print("Database tables initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
    
    def save_message(self, user_id, session_id, role, message):
        """Save chat message to database"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO chat_history (user_id, session_id, role, message)
                VALUES (%s, %s, %s, %s)
            """, (user_id, session_id, role, message))
            
            connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            print(f"Error saving message: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
    
    def get_chat_history(self, user_id, limit=100):
        """Get recent chat history for user"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT session_id, role, message, timestamp
                FROM chat_history
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_user_sessions(self, user_id):
        """Get all chat sessions for a user"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT DISTINCT session_id, 
                       MIN(timestamp) as first_message,
                       MAX(timestamp) as last_message,
                       COUNT(*) as message_count
                FROM chat_history
                WHERE user_id = %s
                GROUP BY session_id
                ORDER BY last_message DESC
            """, (user_id,))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error getting user sessions: {str(e)}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_session_messages(self, user_id, session_id):
        """Get all messages for a specific session"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT role, message, timestamp
                FROM chat_history
                WHERE user_id = %s AND session_id = %s
                ORDER BY timestamp ASC
            """, (user_id, session_id))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error getting session messages: {str(e)}")
            return []
        finally:
            if connection:
                connection.close()
    
    def save_document(self, user_id, session_id, filename, file_type, file_size):
        """Save document metadata to database"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO documents (user_id, session_id, filename, file_type, file_size)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, session_id, filename, file_type, file_size))
            
            connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            print(f"Error saving document: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
