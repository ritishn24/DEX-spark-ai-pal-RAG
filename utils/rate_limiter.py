import time
from collections import defaultdict, deque
from functools import wraps
from flask import request, jsonify, session
import os

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.per_minute_limit = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
        self.per_hour_limit = int(os.getenv('RATE_LIMIT_PER_HOUR', 1000))
    
    def get_client_id(self):
        """Get unique client identifier"""
        if 'user_id' in session:
            return f"user_{session['user_id']}"
        return request.remote_addr
    
    def is_allowed(self, client_id):
        """Check if request is allowed based on rate limits"""
        now = time.time()
        
        # Clean old requests
        self.cleanup_old_requests(client_id, now)
        
        # Check per-minute limit
        minute_requests = [req_time for req_time in self.requests[client_id] if now - req_time < 60]
        if len(minute_requests) >= self.per_minute_limit:
            return False, "Rate limit exceeded: too many requests per minute"
        
        # Check per-hour limit
        hour_requests = [req_time for req_time in self.requests[client_id] if now - req_time < 3600]
        if len(hour_requests) >= self.per_hour_limit:
            return False, "Rate limit exceeded: too many requests per hour"
        
        # Add current request
        self.requests[client_id].append(now)
        return True, None
    
    def cleanup_old_requests(self, client_id, now):
        """Remove requests older than 1 hour"""
        while self.requests[client_id] and now - self.requests[client_id][0] > 3600:
            self.requests[client_id].popleft()

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(f):
    """Decorator to apply rate limiting to routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = rate_limiter.get_client_id()
        allowed, error_message = rate_limiter.is_allowed(client_id)
        
        if not allowed:
            return jsonify({'error': error_message}), 429
        
        return f(*args, **kwargs)
    return decorated_function