from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime


class UserSignupRequest(BaseModel):
    """Request model for user signup"""
    name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @field_validator('password')
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response model for user data (without password)"""
    id: str
    name: str
    email: str
    created_at: datetime


class TokenResponse(BaseModel):
    """Response model for authentication token"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class User(BaseModel):
    """Internal user model with password hash"""
    id: str
    name: str
    email: str
    password_hash: str
    created_at: datetime

