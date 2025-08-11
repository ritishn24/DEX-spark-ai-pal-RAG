
import os
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from .database import DatabaseManager

class AuthManager:
    def __init__(self):
        self.db = DatabaseManager()
    
    def create_user(self, username, password, email=None):
        """Create a new user account"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return False
            
            # Hash password and create user
            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                (username, hashed_password, email)
            )
            connection.commit()
            return True
            
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return False
        finally:
            if connection:
                connection.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute(
                "SELECT id, username, password FROM users WHERE username = %s",
                (username,)
            )
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password'], password):
                return {'id': user['id'], 'username': user['username']}
            
            return None
            
        except Exception as e:
            print(f"Error authenticating user: {str(e)}")
            return None
        finally:
            if connection:
                connection.close()
    
    def get_user_by_id(self, user_id):
        """Get user information by ID"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute(
                "SELECT id, username, email, created_at FROM users WHERE id = %s",
                (user_id,)
            )
            return cursor.fetchone()
            
        except Exception as e:
            print(f"Error getting user by ID: {str(e)}")
            return None
        finally:
            if connection:
                connection.close()
