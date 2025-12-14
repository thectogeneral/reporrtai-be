import os, json
from dotenv import load_dotenv
import re
import asyncio

from datetime import datetime, timedelta
import uuid

# This will search for a .env file in the current directory or parent directories
load_dotenv()

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import praw
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import jwt
from passlib.context import CryptContext
from models.PainPointModel import PainPointOutput
from models.AppIdeaModel import AppIdeaOutput
from models.PerformanceReportModel import PerformanceReportOutput
from models.SentimentModel import SentimentExtractionOutput
from models.TopicModel import TopicOutput
from models.IdeaTopicModel import IdeaTopicOutput
from models.UserModel import (
    UserSignupRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    User
)
from models.UserDBModel import UserDB
from database import get_db, init_db, close_db, async_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


# ========== JWT Configuration ==========

# Secret key for JWT encoding/decoding - CHANGE THIS IN PRODUCTION!
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()

# ========== Rate Limiting Configuration ==========
def get_user_id_from_request(request: Request) -> str:
    """Extract user ID from JWT token for rate limiting"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload.get("sub", get_remote_address(request))
        except:
            pass
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_id_from_request)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token"""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT access token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserDB:
    """Dependency to get the current authenticated user from JWT token"""
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    
    result = await db.execute(select(UserDB).where(UserDB.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[UserDB]:
    """Find a user by email address"""
    result = await db.execute(select(UserDB).where(UserDB.email == email))
    return result.scalar_one_or_none()


# ========== Reddit Fetching ==========

def parse_reddit_url(url: str) -> Dict[str, str]:
    """
    Extract subreddit, post_id from a Reddit thread URL.
    https://www.reddit.com/r/DataHoarder/comments/1k8wnza/with_the_rate_limiting_everywhere_does_anyone/
    E.g. https://www.reddit.com/r/indiehackers/comments/1o10kjh/... → post_id = "1o10kjh"
    """
    parsed = urlparse(url)
    parts = parsed.path.split('/')
    # Expected pattern: /r/<subreddit>/comments/<post_id>/...
    try:
        i = parts.index("comments")
        subreddit = parts[i - 1]
        post_id = parts[i + 1]
    except ValueError:
        raise ValueError("URL not in expected format")
    return {"subreddit": subreddit, "post_id": post_id}

def fetch_reddit_thread(reddit, url: str, limit_comments: int = 30) -> str:
    """
    Returns a full text (post + some comments) as one big string.
    """
    info = parse_reddit_url(url)
    subreddit = info["subreddit"]
    post_id = info["post_id"]
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)

    texts = []
    # original post
    texts.append(f"Title: {submission.title}\n")
    texts.append(submission.selftext + "\n\n")

    # comments
    count = 0
    for comment in submission.comments.list():
        if count >= limit_comments:
            break
        texts.append(f"Comment by {comment.author}: {comment.body}\n")
        count += 1

    return "\n".join(texts)



# ========== LLM Chains for Extraction & Idea ==========

idea_multiple_single = """You are an innovative product strategist and system designer. 
        Given these pain points:
        {pain_points}

        PRODUCT STRATEGY SECTIONS TO GENERATE FOR EACH APP IDEA:
            1. Product Overview
            2. Industry
            3. Target Users
            4. Competitive Landscape
            5. Core Features (MVP + later versions)
            6. Mapping Between Features and Pain Points
            7. MVP Scope
            8. Product Roadmap (MVP → V1 → V2 → Full Product)
            9. UX & Design Considerations
            10. Technical Implementation Overview
            11. Business Model
            12. Monetization Strategies
            13. Go-To-Market Strategy
            14. Success Metrics
            15. Challenges & Risks
            16. Team & Skills Required
            17. Long-Term Opportunities & 5-Year Vision

            You must fill every section. If information is missing, make reasonable assumptions."""

idea_topic_template = """You are an innovative product strategist and system designer. 
        Given this context:

        Thread text:
        {thread_text}

        Pain points:
        {pain_points}

        Generate 3-5 unique, actionable app ideas that directly address the given pain points. Each idea should be conceptually distinct from all others.

        PRODUCT STRATEGY SECTIONS TO GENERATE FOR EACH APP IDEA:
            1. App Name
            2. Tagline
            3. Description
            4. Pain Points Solved

        You must fill every section. If information is missing, make reasonable assumptions.
        """

painpoint_template = """You are an expert product analyst. Analyze the conversation/thread text below and extract the main user pain points, then produce a comprehensive product strategy. Follow the rules EXACTLY.

            CRITICAL EXTRACTION RULES:
                1. Extract only real and meaningful pain points from the actual text.
                2. For EACH pain point include:
                    - title
                    - number_of_users
                    - category
                    - quotes (array of quotes)
                3. Quotes must be copied EXACTLY — no paraphrasing.
                4. Include username attribution when available.
                5. If multiple users mention the same issue, include multiple quotes.
                6. If the pain point is implied (not quoted), use: "quote": "Implicit, not explicitly quoted".
                7. You MUST return at least 3 pain points. If the text has fewer than 3, merge smaller related issues into distinct categories.
                8. After extraction, propose 1(one) app ideas that directly address these issues.
                9. Then generate a complete product vision strategy following all sections defined below.
                10. The ENTIRE output MUST be valid JSON. No markdown, no commentary.

            
            Thread text:
            {thread_text}
            """



topic_template = """You are an expert product analyst. Analyze the conversation/thread text below and extract all *relevant topics* discussed. Follow the rules EXACTLY.

        TOPIC EXTRACTION RULES:
            1. Extract only meaningful topics that are directly discussed or implied in the text.
            2. For EACH topic include:
                - topic (a concise, descriptive label of the topic)
                - subtopics (if applicable, a list of related subtopics or aspects discussed under this topic; otherwise an empty list)
                - brief_reason (1 short sentence explaining why this topic is relevant in the thread)
            3. Do NOT combine separate topics into one.
            4. If a topic is implied but not explicitly stated, set:
                "topic": "Implicit, not explicitly mentioned"
            5. Include at least 3 topics. If the thread has fewer, break down discussions into smaller topic elements.
            6. The ENTIRE output MUST be valid JSON. No markdown, no commentary.

        Thread text:
        {thread_text}
        """


performance_review_template = """You are a strategic product analyst and business reviewer. 
        Given this context:
        {thread_text}

        You must fill every section. If information is missing, make reasonable assumptions.
        Your response should read like a **professional product performance report** — data-driven, reflective, and actionable.

        
        PERFORMANCE REPORT SECTIONS TO GENERATE:
            1. Executive Summary
            2. Customer Insights & Pain Points
            3. Product Performance
            4. Business & Market Performance
            5. Operational Review
            6. Marketing & Growth Review
            7. Technology & Infrastructure Review
            8. Financial Overview
            9. Lessons Learned
            10. Strategic Adjustments & Next Steps
            11. Future Roadmap
            12. Risks & Mitigation
            13. Conclusion
        """


sentiment_template = """You are an expert product analyst. Analyze the conversation/thread text below and extract all *relevant user sentiments*. Follow the rules EXACTLY.

        SENTIMENT EXTRACTION RULES:
            1. Extract only meaningful sentiments directly expressed in the text.
            2. For EACH sentiment include:
                - user (the username of the user who wrote the quote)
                - sentiment (one of: "positive", "negative", "neutral")
                - quote (MUST be copied EXACTLY from the thread — no paraphrasing)
                - brief_reason (1 short sentence explaining why this quote fits the sentiment)
            3. Quotes must be copied EXACTLY — no paraphrasing.
            4. Do NOT merge quotes — each unique user quote should become its own sentiment item.
            5. Include username attribution when available.
            6. If the sentiment is implied but not explicitly stated, set:
                "quote": "Implicit, not explicitly quoted"
            7. Minimum of 3 sentiments. If the thread has fewer, break down comments into smaller sentiment expressions.
            8. The ENTIRE output MUST be valid JSON. No markdown, no commentary.

        Thread text:
        {thread_text}
        """




# ========== Main Flow ==========

def make_sentiment_chain(llm):
    prompt = PromptTemplate(input_variables=["thread_text"], template=sentiment_template)
    return prompt | llm

def make_painpoint_chain(llm):
    prompt = PromptTemplate(input_variables=["thread_text"], template=painpoint_template)
    return prompt | llm

def make_idea_topic_chain(llm):
    prompt = PromptTemplate(input_variables=["thread_text", "pain_points"], template=idea_topic_template)
    return prompt | llm

def make_topic_chain(llm):
    prompt = PromptTemplate(input_variables=["thread_text"], template=topic_template)
    return prompt | llm

def make_performance_review_chain(llm):
    prompt = PromptTemplate(input_variables=["thread_text"], template=performance_review_template)
    return prompt | llm


def make_idea_chain(llm):
    prompt = PromptTemplate(input_variables=["pain_points"], template=idea_multiple_single)
    return prompt | llm

def choose_llm(use_json_mode: bool = True, output_model: str = PainPointOutput):
    """Try Ollama first; if fails, fallback to OpenAI
    
    Args:
        use_json_mode: If True, use JSON format for Ollama (format="json")
    """
    # You may check some env var to force which you want
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() in ("1", "true", "yes")
    # Lower temperature for more deterministic JSON output
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    # Increase max tokens to avoid truncation
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    
    
    if use_ollama:
        try:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
            
            # Use ChatOllama - defaults to http://localhost:11434
            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            # Add format="json" if use_json_mode is True
            if use_json_mode:
                llm_kwargs["format"] = "json"
                print("DEBUG: Using Ollama with format='json'")
            
            print(f"DEBUG: Connecting to Ollama with model {model_name}")
            
            llm = ChatOllama(**llm_kwargs).with_structured_output(output_model)
            return llm
        except Exception as e:
            error_msg = str(e)
            print(f"Ollama init failed: {error_msg}")
            print("Falling back to OpenAI...")
    # Fallback to OpenAI
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    llm = ChatOpenAI(
        temperature=temperature,
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens
    ).with_structured_output(output_model)

    return llm

def validate_link(url: str) -> str:
    try:
        # Check if it's a Reddit link
        reddit_pattern = r'https?://(?:www\.)?reddit\.com/r/.*?/comments/.*'
        if re.match(reddit_pattern, url):
            return generate_reddit_report(url)
        
        # Check if it's an X (Twitter) link
        x_pattern = r'https?://(?:www\.)?(?:twitter\.com|x\.com)/.*'
        if re.match(x_pattern, url):
            return "X link detected - no action taken"

        raise ValueError("invalid link")
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in valide_link: {str(e)}")
        return f"Error processing link: {str(e)}"

def generate_reddit_report(url: str) -> str:
    """
    Generate a report from a Reddit thread with error handling.
    Returns a formatted string (for backward compatibility).
    """
    result = generate_reddit_report_structured(url)
    if isinstance(result, dict) and "error" in result:
        return result["error"]
    
    report = []
    report.append("=== Pain Points ===\n")
    report.append(result["pain_points"].strip())
    report.append("\n\n=== App Idea & Solution ===\n")
    report.append(result["app_idea"].strip())
    return "\n".join(report)





# ========== Helper Functions for Chain Invocation ==========

def _extract_chain_response(result: Any, fallback_key: str = "") -> str:
    """
    Extract content from a chain invocation result.
    Handles AIMessage objects, dictionaries, and other types.
    """
    if hasattr(result, 'content'):
        return result.content
    elif isinstance(result, dict):
        return result.get(fallback_key, "")
    else:
        return str(result)

def _handle_ollama_403_error(error: Exception) -> Optional[Dict[str, str]]:
    """
    Handle 403 errors from Ollama API requests.
    Returns an error dictionary if it's a 403 error, None otherwise.
    """
    error_msg = str(error)
    error_type = type(error).__name__
    
    is_403_error = (
        "403" in error_msg or 
        "status code: 403" in error_msg.lower() or 
        error_type == "HTTPStatusError"
    )
    
    if not is_403_error:
        return None
    
    use_ollama = os.getenv("USE_OLLAMA", "true").lower()
    error_details = f"""Error: Ollama API request failed (403)

        Configuration:
        - USE_OLLAMA: {use_ollama}

        Possible solutions:
        1. Check if Ollama is running: `ollama serve` or `curl http://localhost:11434/api/tags`
        2. Check if your Ollama instance is accessible
        3. Try setting USE_OLLAMA=false in .env to use OpenAI instead

        Error details: {error_msg}
        Error type: {error_type}"""
    return {"error": error_details.strip()}

def _invoke_chain_safely(chain, input_data: Dict[str, str], fallback_key: str = "") -> Any:
    """
    Safely invoke a chain. Returns the structured Pydantic object if available,
    otherwise falls back to string extraction.
    """
    try:
        result = chain.invoke(input_data)
        # If result is Pydantic model, return it directly
        if isinstance(result, BaseModel):
            return result
        return _extract_chain_response(result, fallback_key)
    except Exception as e:
        error_response = _handle_ollama_403_error(e)
        if error_response:
            raise ValueError(error_response["error"])
        raise

def generate_reddit_report_structured(url: str) -> Dict[str, Any]:
    """
    Generate a structured report from a Reddit thread with error handling.
    Returns a dictionary with separate fields: pain_points, app_idea, performance_review.
    """
    try:
        # Initialize Reddit client with error handling
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "app-idea-generator"
        )
        
        use_json_mode = os.getenv("USE_JSON", "true").lower() in ("1", "true", "yes")

        # Test Reddit connection
        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}
        
        thread_text = fetch_reddit_thread(reddit, url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        pain_llm = choose_llm(use_json_mode=use_json_mode, output_model=PainPointOutput)  # Use text mode for formatted output
        pain_chain = make_painpoint_chain(pain_llm)

        idea_llm = choose_llm(use_json_mode=use_json_mode, output_model=AppIdeaOutput)
        idea_chain = make_idea_chain(idea_llm)

        performance_llm = choose_llm(use_json_mode=use_json_mode, output_model=PerformanceReportOutput)
        performance_chain = make_performance_review_chain(performance_llm)

        # Invoke chains with error handling
        try:
            pain_resp = _invoke_chain_safely(
                pain_chain, 
                {"thread_text": thread_text}, 
                fallback_key="pain_points"
            )
        except ValueError as e:
            return {"error": str(e)}

        try:
            idea_resp = _invoke_chain_safely(
                idea_chain, 
                {"pain_points": pain_resp}, 
                fallback_key="app_idea"
            )
        except ValueError as e:
            return {"error": str(e)}

        try:
            performance_resp = _invoke_chain_safely(
                performance_chain, 
                {"thread_text": thread_text}, 
                fallback_key="performance_report"
            )
        except ValueError as e:
            return {"error": str(e)}

        return {
            "pain_points": pain_resp.strip(),
            "app_idea": idea_resp.strip(),
            "performance_review": performance_resp.strip(),
            "url": url
        }
        
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in generate_reddit_report_structured: {str(e)}")
        return {"error": f"Error generating report: {str(e)}"}

def save_report(report: str, filename: str = "report.txt"):
    # This will create the file in the **current working directory**
    cwd = os.getcwd()
    fullpath = os.path.join(cwd, filename)
    with open(fullpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {fullpath}")





# ========== FastAPI Application ==========

app = FastAPI(title="Reporrt AI API", description="API for generating reports from Reddit threads")

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global chains reused across requests
performance_chain_global = None
sentiment_chain_global = None
topic_chain_global = None
pain_chain_global = None
idea_chain_global = None
idea_topic_chain_global = None

# In-memory progress tracking for long-running requests
progress_store: Dict[str, Dict[str, Any]] = {}


def _set_progress(request_id: Optional[str], progress: int, status: str) -> None:
    """
    Update progress percentage and status text for a given request_id.
    Safe to call from within the main async context; not used inside worker threads.
    """
    if not request_id:
        return
    progress_store[request_id] = {
        "progress": max(0, min(100, progress)),
        "status": status,
    }


# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global performance_chain_global, sentiment_chain_global, topic_chain_global
    global pain_chain_global, idea_chain_global, idea_topic_chain_global

    # Initialize database tables
    print("Initializing database...")
    await init_db()
    print("Database initialized successfully")

    use_json_mode = os.getenv("USE_JSON", "true").lower() in ("1", "true", "yes")

    # Create each LLM only once
    performance_llm = choose_llm(use_json_mode=use_json_mode, output_model=PerformanceReportOutput)
    sentiment_llm = choose_llm(use_json_mode=use_json_mode, output_model=SentimentExtractionOutput)
    topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=TopicOutput)
    pain_llm = choose_llm(use_json_mode=use_json_mode, output_model=PainPointOutput)
    idea_llm = choose_llm(use_json_mode=use_json_mode, output_model=AppIdeaOutput)
    idea_topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=IdeaTopicOutput)

    # And their chains
    performance_chain_global = make_performance_review_chain(performance_llm)
    sentiment_chain_global = make_sentiment_chain(sentiment_llm)
    topic_chain_global = make_topic_chain(topic_llm)
    pain_chain_global = make_painpoint_chain(pain_llm)
    idea_chain_global = make_idea_chain(idea_llm)
    idea_topic_chain_global = make_idea_topic_chain(idea_topic_llm)


@app.on_event("shutdown")
async def shutdown():
    """Close database connections on shutdown"""
    print("Closing database connections...")
    await close_db()
    print("Database connections closed")

@app.get("/")
async def root():
    return {"message": "Reporrt AI API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


# ========== Authentication Endpoints ==========

@app.post("/api/v1/auth/signup", response_model=TokenResponse)
async def signup(request: UserSignupRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user account.
    
    - **name**: User's full name
    - **email**: User's email address (must be unique)
    - **password**: Password (minimum 8 characters)
    - **confirm_password**: Must match password
    """
    # Check if passwords match
    if request.password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Check if email already exists
    existing_user = await get_user_by_email(db, request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    now = datetime.utcnow()
    
    new_user = UserDB(
        name=request.name,
        email=request.email,
        password_hash=hash_password(request.password),
        created_at=now
    )
    
    # Store user in database
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Generate access token
    access_token = create_access_token(str(new_user.id), request.email)
    
    # Return token and user info
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(new_user.id),
            name=new_user.name,
            email=new_user.email,
            created_at=new_user.created_at
        )
    )


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user and return an access token.
    
    - **email**: User's email address
    - **password**: User's password
    """
    # Find user by email
    user = await get_user_by_email(db, request.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate access token
    access_token = create_access_token(str(user.id), user.email)
    
    # Return token and user info
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(user.id),
            name=user.name,
            email=user.email,
            created_at=user.created_at
        )
    )


@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_me(current_user: UserDB = Depends(get_current_user)):
    """
    Get the current authenticated user's information.
    Requires a valid JWT token in the Authorization header.
    """
    return UserResponse(
        id=str(current_user.id),
        name=current_user.name,
        email=current_user.email,
        created_at=current_user.created_at
    )


@app.post("/api/v1/idea")
@limiter.limit("5/day")
async def generate_idea_endpoint(
    request: Request,
    url: str,
    request_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user)
):
    """Generate app idea from subreddit URL. Limited to 5 requests per day."""
    return await generate_idea(url, request_id)

@app.post("/api/v1/performance")
@limiter.limit("5/day")
async def generate_performance_report_endpoint(
    request: Request,
    url: str,
    request_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user)
):
    """Generate performance report from subreddit URL. Limited to 5 requests per day."""
    return await generate_performance_report(url, request_id)

@app.get("/api/v1/progress")
async def get_progress(request_id: str):
    """
    Poll current progress of a long-running request.
    Frontend should:
      1) Generate a request_id (e.g. UUID)
      2) Pass it to /idea or /performance
      3) Poll this endpoint with the same request_id
    """
    data = progress_store.get(request_id)
    if not data:
        return {"request_id": request_id, "progress": 0, "status": "unknown"}
    return {"request_id": request_id, **data}

async def generate_idea(url: str, request_id: Optional[str] = None):
    try:
        global pain_chain_global, idea_chain_global, idea_topic_chain_global

        _set_progress(request_id, 0, "starting")

        # Initialize Reddit client with error handling
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "app-idea-generator"
        )
        
        use_json_mode = os.getenv("USE_JSON", "true").lower() in ("1", "true", "yes")

        # Test Reddit connection
        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}
        
        # Run blocking Reddit fetch in a thread so it doesn't block the event loop
        thread_text = await asyncio.to_thread(fetch_reddit_thread, reddit, url)
        _set_progress(request_id, 20, "fetched thread content")
        
        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        # Safety fallback: if startup failed for some reason, init lazily
        if not all([pain_chain_global, idea_chain_global, idea_topic_chain_global]):
            pain_llm = choose_llm(use_json_mode=use_json_mode, output_model=PainPointOutput)
            idea_llm = choose_llm(use_json_mode=use_json_mode, output_model=AppIdeaOutput)
            idea_topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=IdeaTopicOutput)
            pain_chain_global = make_painpoint_chain(pain_llm)
            idea_chain_global = make_idea_chain(idea_llm)
            idea_topic_chain_global = make_idea_topic_chain(idea_topic_llm)

        # First get pain points
        try:
            pain_resp = await asyncio.to_thread(
                _invoke_chain_safely,
                pain_chain_global,
                {"thread_text": thread_text},
                "pain_points",
            )
        except ValueError as e:
            return {"error": str(e)}
        _set_progress(request_id, 50, "generated pain points")

        # Then derive idea + idea_topic in parallel from the same pain points
        try:
            idea_topic_resp, idea_resp = await asyncio.gather(
                asyncio.to_thread(
                    _invoke_chain_safely,
                    idea_topic_chain_global,
                    {"pain_points": pain_resp, "thread_text": thread_text},
                    "product_ideas",
                ),

                _set_progress(request_id, 75, "generated idea topics"),

                asyncio.to_thread(
                    _invoke_chain_safely,
                    idea_chain_global,
                    {"pain_points": pain_resp},
                    "app_idea",
                ),
            )
        except ValueError as e:
            return {"error": str(e)}

        _set_progress(request_id, 95, "assembled ideas")

        return {
            "pain_points": pain_resp,
            "idea_resp": idea_resp,
            "idea_topic_resp": idea_topic_resp,
            "url": url,
            "request_id": request_id,
        }
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in generate reddit idea: {str(e)}")
        return {"error": f"Error generating report: {str(e)}"}

async def generate_performance_report(url: str, request_id: Optional[str] = None):
    try:
        global performance_chain_global, sentiment_chain_global, topic_chain_global

        _set_progress(request_id, 0, "starting")

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "app-idea-generator",
        )

        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}

        thread_text = await asyncio.to_thread(fetch_reddit_thread, reddit, url)
        _set_progress(request_id, 20, "fetched thread content")

        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        # Safety fallback: if startup failed for some reason, init lazily
        if not all([performance_chain_global, sentiment_chain_global, topic_chain_global]):
            use_json_mode = os.getenv("USE_JSON", "true").lower() in ("1", "true", "yes")
            performance_llm = choose_llm(use_json_mode=use_json_mode, output_model=PerformanceReportOutput)
            sentiment_llm = choose_llm(use_json_mode=use_json_mode, output_model=SentimentExtractionOutput)
            topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=TopicOutput)
            performance_chain_global = make_performance_review_chain(performance_llm)
            sentiment_chain_global = make_sentiment_chain(sentiment_llm)
            topic_chain_global = make_topic_chain(topic_llm)

        # Use the shared chains (still invoked concurrently, as you already do)
        try:
            topic_resp, sentiment_resp, performance_resp = await asyncio.gather(
                asyncio.to_thread(
                    _invoke_chain_safely,
                    topic_chain_global,
                    {"thread_text": thread_text},
                    "topics",
                ),
                asyncio.to_thread(
                    _invoke_chain_safely,
                    sentiment_chain_global,
                    {"thread_text": thread_text},
                    "sentiments",
                ),
                asyncio.to_thread(
                    _invoke_chain_safely,
                    performance_chain_global,
                    {"thread_text": thread_text},
                    "performance_report",
                ),
            )
        except ValueError as e:
            return {"error": str(e)}

        _set_progress(request_id, 95, "completed analysis")

        return {
            "performance_review": performance_resp,
            "sentiments": sentiment_resp,
            "topics": topic_resp,
            "url": url,
            "request_id": request_id,
        }
    except Exception as e:
        print(f"Error in generate generating performance report: {str(e)}")
        return {"error": f"Error generating report: {str(e)}"}


if __name__ == "__main__":
    import sys
    import socket
    
    # Check if running as API server
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        try:
            import uvicorn
            
            def find_free_port(start_port=8000):
                """Find a free port starting from start_port"""
                port = start_port
                while port < start_port + 100:  # Limit search to avoid infinite loop
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        try:
                            s.bind(('', port))
                            return port
                        except OSError:
                            port += 1
                raise RuntimeError(f"Could not find a free port starting from {start_port}")
            
            port = int(os.getenv("PORT", 0))
            if port == 0:
                port = find_free_port(8000)
            
            print(f"Starting server on http://0.0.0.0:{port}")
            uvicorn.run(app, host="0.0.0.0", port=port)
        except ImportError:
            print("Error: uvicorn is not installed. Please run: pip install uvicorn")
            print("Or install all requirements: pip install -r requirements.txt")
            sys.exit(1)
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"Error: Port {port} is already in use.")
                print(f"Please stop the process using port {port} or set PORT environment variable to a different port.")
                print(f"Example: PORT=8001 python main.py serve")
                sys.exit(1)
            raise
    else:
        # CLI mode (original functionality)
        try:
            url = input("Enter Reddit thread URL: ").strip()
            if not url:
                print("Error: No URL provided")
                exit(1)
                
            report_text = validate_link(url)
            output_file = "report.txt"
            save_report(report_text, output_file)
            print(f"Report saved to {output_file}")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print("Please check your input and try again")











