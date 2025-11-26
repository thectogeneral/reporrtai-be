import os, json
from dotenv import load_dotenv
import re


# This will search for a .env file in the current directory or parent directories
load_dotenv()

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import praw
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.PainPointModel import PainPointOutput
from models.AppIdeaModel import AppIdeaOutput
from models.PerformanceReportModel import PerformanceReportOutput
from models.SentimentModel import SentimentExtractionOutput
from models.TopicModel import TopicOutput
from models.IdeaTopicModel import IdeaTopicOutput



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

def fetch_reddit_thread(reddit, url: str, limit_comments: int = 100) -> str:
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


performance_review_template = """You are a strategic product analyst and business reviewer. 
        Given this context:
        {thread_text}

        Generate a comprehensive performance report on the product or company, focusing on results, lessons, and future directions. Your response must fill every section. If information is missing, make reasonable and realistic assumptions,
        
        Your report should cover the following sections:

        1. **Executive Summary**
        - Brief overview of the product or company
        - Key goals or KPIs initially set
        - Short summary of the overall performance (successes and challenges)

        2. **Customer Insights & Pain Points**
        - What user pain points were identified at launch?
        - How effectively were they solved?
        - What new pain points or unmet needs have emerged?
        - Include user feedback highlights or sentiment trends.

        3. **Product Performance**
        - What worked well (features, user experience, engagement)?
        - What didn’t work or underperformed (features, design choices, functionality)?
        - Adoption metrics, engagement data, or usage trends.

        4. **Business & Market Performance**
        - Revenue performance vs. projections
        - Customer acquisition and retention analysis
        - Market share and competitive positioning

        5. **Operational Review**
        - Internal execution and workflow effectiveness
        - Bottlenecks or inefficiencies
        - Team culture and alignment

        6. **Marketing & Growth Review**
        - Performance of marketing and acquisition channels
        - Conversion funnel analysis
        - Brand perception and community engagement

        7. **Technology & Infrastructure Review**
        - Stability, uptime, scalability
        - Key bugs, performance bottlenecks, or technical debt
        - Architecture or tech-stack lessons learned

        8. **Financial Overview**
        - Revenue and cost breakdown
        - Profitability trends and key financial metrics

        9. **Lessons Learned**
        - Key wins, mistakes, and insights
        - Assumptions that proved right or wrong

        10. **Strategic Adjustments & Next Steps**
            - Planned improvements and pivots
            - Areas to deprioritize or sunset

        11. **Future Roadmap**
            - Short-term (3–6 months): Key fixes or goals
            - Mid-term (6–12 months): Growth and optimization
            - Long-term (1–3 years): Vision and scaling goals

        12. **Risks & Mitigation**
            - Key forward risks
            - Mitigation strategies

        13. **Conclusion**
            - Summary of momentum and outlook
            - Strategic recommendations for next phase

        You must fill every section. If information is missing, make reasonable assumptions.
        Your response should read like a **professional product performance report** — data-driven, reflective, and actionable."""


sentiment_template = """You are an expert product analyst. Analyze the conversation/thread text below and extract all *relevant user sentiments*. Follow the rules EXACTLY.

        CRITICAL RULES:
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

        CRITICAL RULES:
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




import json
from typing import Union

def pretty_json(response_str: str) -> Union[str, None]:
    """
    Convert a raw JSON string (with escaped characters) into
    pretty-printed, readable JSON.

    Args:
        response_str (str): The raw JSON string.

    Returns:
        str: Pretty-printed JSON string, or None if invalid.
    """
    try:
        # Convert string to Python dict/list
        response_dict = json.loads(response_str)
        # Return pretty-printed JSON
        return json.dumps(response_dict, indent=2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None



# ========== FastAPI Application ==========

app = FastAPI(title="Reporrt AI API", description="API for generating reports from Reddit threads")

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Reporrt AI API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/idea")
async def generate_idea(url: str):
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

        #idea_llm = choose_llm(use_json_mode=use_json_mode, output_model=AppIdeaOutput)
        #idea_chain = make_idea_chain(idea_llm)

        idea_topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=IdeaTopicOutput)
        idea_topic_chain = make_idea_topic_chain(idea_topic_llm)

       

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
            idea_topic_resp = _invoke_chain_safely(
                idea_topic_chain, 
                {"pain_points": pain_resp, "thread_text": thread_text}, 
                fallback_key="product_ideas"
            )
        except ValueError as e:
            return {"error": str(e)}

        #try:
            #idea_resp = _invoke_chain_safely(
                #idea_chain, 
                #{"pain_points": pain_resp}, 
                #fallback_key="app_idea"
            #)
        #except ValueError as e:
            #return {"error": str(e)}

        return {
            #"pain_points": pain_resp,
            #"idea_resp": idea_resp,
            "idea_topic_resp": idea_topic_resp,
            #"url": url
        }
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in generate reddit idea: {str(e)}")
        return {"error": f"Error generating report: {str(e)}"}

@app.get("/api/v1/performance")
async def generate_performance_report(url: str):
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
            
        performance_llm = choose_llm(use_json_mode=use_json_mode, output_model=PerformanceReportOutput)
        performance_chain = make_performance_review_chain(performance_llm)

        sentiment_llm = choose_llm(use_json_mode=use_json_mode, output_model=SentimentExtractionOutput)
        sentiment_chain = make_sentiment_chain(sentiment_llm)

        topic_llm = choose_llm(use_json_mode=use_json_mode, output_model=TopicOutput)
        topic_chain = make_topic_chain(topic_llm)

        try:
            topic_resp = _invoke_chain_safely(
                topic_chain, 
                {"thread_text": thread_text}, 
                fallback_key="topics"
            )
        except ValueError as e:
            return {"error": str(e)}

        try:
            sentiment_resp = _invoke_chain_safely(
                sentiment_chain, 
                {"thread_text": thread_text}, 
                fallback_key="sentiments"
            )
        except ValueError as e:
            return {"error": str(e)}

        # Invoke chains with error handling
        try:
            performance_resp = _invoke_chain_safely(
                performance_chain, 
                {"thread_text": thread_text}, 
                fallback_key="performance_report"
            )
        except ValueError as e:
            return {"error": str(e)}

        return {
            "performance_review": performance_resp,
            "sentiments": sentiment_resp,
            "topics": topic_resp,
            "url": url
        }
    except Exception as e:
        # Log the error and return a safe error message
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











