"""Authentication endpoints for login and token management.

Provides login endpoint that validates API keys and returns JWT tokens.
"""

from fastapi import HTTPException, status
from pydantic import BaseModel
from src.core.auth import jwt_auth


class LoginRequest(BaseModel):
    """Request model for login."""
    api_key: str


class LoginResponse(BaseModel):
    """Response model for successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class AuthEndpoints:
    """Authentication endpoints."""
    
    @staticmethod
    async def login(request: LoginRequest):
        """Login with API key and receive JWT token.
        
        Args:
            request: LoginRequest containing the API key
            
        Returns:
            LoginResponse with JWT token
            
        Raises:
            HTTPException: If API key is invalid
        """
        try:
            # Create JWT token (validates API key internally)
            token = jwt_auth.create_token(request.api_key)
            
            return LoginResponse(
                access_token=token,
                token_type="bearer",
                expires_in=1800  # 30 minutes in seconds
            )
        except HTTPException:
            # Re-raise HTTP exceptions from token creation
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Login failed: {str(e)}"
            )
    
    @staticmethod
    async def verify_token():
        """Verify current token (protected endpoint).
        
        This endpoint requires a valid JWT token and can be used
        to check if the current token is still valid.
        
        Returns:
            Simple success message if token is valid
        """
        return {"status": "valid", "message": "Token is valid"} 