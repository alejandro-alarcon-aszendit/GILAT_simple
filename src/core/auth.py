"""Authentication middleware for the Document Service.

Provides JWT-based authentication with login endpoint and token validation.
"""

import os
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional


class JWTAuth:
    """JWT-based authentication handler."""
    
    def __init__(self):
        self.api_key = os.getenv("API_AUTH_KEY")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.security = HTTPBearer(auto_error=False)
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login"  # Login endpoint is public
        }
    
    def create_token(self, api_key: str) -> str:
        """Create a JWT token for authenticated users.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            JWT token string
            
        Raises:
            HTTPException: If API key is invalid
        """
        if not self.api_key:
            # If no API key is configured, allow access
            payload = {
                "sub": "public_access",
                "exp": datetime.utcnow() + timedelta(minutes=30),
                "iat": datetime.utcnow()
            }
            return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        if api_key != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Create JWT token
        payload = {
            "sub": "authenticated_user",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> bool:
        """Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return True
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Verify JWT token authentication.
        
        Args:
            request: FastAPI request object
            
        Returns:
            None if authenticated or endpoint is public
            
        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return None
            
        # Skip authentication if no API key is configured
        if not self.api_key:
            return None
            
        # Get authorization header
        authorization: HTTPAuthorizationCredentials = await self.security(request)
        
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required. Please login first.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify JWT token
        if not self.verify_token(authorization.credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token. Please login again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return authorization.credentials


# Global auth instance
jwt_auth = JWTAuth() 