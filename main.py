import os
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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

# ========== Pydantic Models for Structured Output ==========

class PainPointModel(BaseModel):
    """Model for a single pain point"""
    description: str = Field(description="The pain point description")
    sentiment: str = Field(description="The sentiment, must be one of: negative, positive, or neutral")
    quote: str = Field(description="A short quote from the text, 1-2 sentences")

class PainPointsListModel(BaseModel):
    """Model for a list of pain points"""
    pain_points: List[PainPointModel] = Field(description="List of pain points extracted from the discussion")

# ========== LLM Chains for Extraction & Idea ==========

def make_painpoint_chain(llm):
    """Create a chain for pain points extraction with formatted output"""
    # Enhanced prompt with clear instructions for formatted text output
    prompt = PromptTemplate(
        template="""You are an expert product analyst. Analyze the conversation/thread text below and extract the main **user pain points** being discussed.

            CRITICAL REQUIREMENTS:
            1. Extract **real, meaningful** pain points from the actual text. Do NOT produce vague or empty descriptions.
            2. For EACH pain point, you MUST include:
                - A short title summarizing the pain point.
                - The **number of unique users** who mentioned or implied this pain point.
                - At least one **direct quote** (1–2 sentences) from the text, copied EXACTLY.
            3. Quotes must be copied **exactly** from the thread text — no paraphrasing.
            4. When a username/author is available, include attribution (e.g., *"— Username"*).
            5. If multiple users expressed the same pain point, include multiple quotes and count them.
            6. If a pain point is inferred but not explicitly stated, label the quote section as: *(Implicit, not explicitly quoted)*.
            7. **MANDATORY:** Return **at least 3 pain points** — ideally 5–10. Never return fewer than 3.
            8. Final output must be formatted as bullet points:
                - Start with a bullet (•)
                - Pain point title in **bold**
                - Followed by: *(X users complained)*   
                - Then list the quotes under an indented block.

            ===== EXAMPLE (DO NOT INCLUDE IN OUTPUT) =====
            • **High withdrawal fees** *(2 users complained)*: 
                "Coinbase charges $906 for the fee..." — Beardog907
                "Coinbase overcharges for withdrawals." — Blacknoyzz

            • **Uncertainty about fees** *(2 users complained)*:
                "What is the justification for cb charging this much?" — 7venhigh
                "Coinbase charges exorbitant fees..." — Whitenoyzz
            ===== END OF EXAMPLE =====
            Thread text:
            {thread_text}
            

            Now extract the pain points from the above text, following the rules above.""",
            input_variables=["thread_text"]
        )
    
    # Simple chain - no JSON format needed for formatted text output
    chain = prompt | llm
    return chain

# Prompt for idea generation (multiple ideas with structured data)
idea_template = """You are an innovative product strategist and system designer. 
        Given these pain points:
        {pain_points}

        Propose **3-5** app ideas or digital solutions that address these issues. For each idea, provide:

        1. **Tagline**: A catchy one-line tagline (max 15 words)
        2. **Problem**: What specific problem(s) does this solution solve? (2-3 sentences)
        3. **Product Description**: A clear description of what the product is and how it works (3-5 sentences)
        4. **Full Detail Report**: A comprehensive product roadmap covering:
        - Product Overview (name, description, value proposition)
        - Industry and target audience
        - Core features (3-7 features)
        - MVP scope
        - Business model
        - Go-to-market strategy
        - Technical implementation overview

        Format your response as readable text with clear sections and formatting. Use markdown formatting for headers, bullet points, and emphasis."""

# Prompt for single idea generation (for backward compatibility)
idea_template_single = """You are an innovative product strategist and system designer. 
        Given these pain points:
        {pain_points}

        Propose **one** app idea or digital solution (name it) that directly addresses these issues. 
        Then provide a **comprehensive and actionable product roadmap** covering the following:

        1. **Product Overview**
        - App Name
        - One-sentence description
        - Problem Summary
        - **Value Proposition Summary:** State clearly how the solution stands out from competitors and what makes it unique or indispensable.
        - Vision and long-term goal

        2. **Industry**
        - What industry is the app in?
        - What is the problem that the app solves?
        - What is the target audience?
        - What is the business stage?

        3. **Target Users**
        - Who are the primary and secondary users?
        - What motivates them?
        - Key demographics and behavioral traits

        4. **Competitive Landscape**
        - Compare the proposed solution to 2–3 existing competitors using a short table or summary.
        - Highlight unique differentiators and market gaps.

        5. **Core Features (3–7)**
        - List and explain each feature.
        - Indicate which features belong in the MVP vs. later versions.

        6. **How It Solves the Pain Points**
        - Clearly map each feature to one or more pain points.
        - Show how these connections create tangible user value.

        7. **MVP (Minimum Viable Product) Scope**
        - What is the smallest version that delivers value?
        - Core workflows or user journeys included.
        - What is the unique selling point (USP) of the app?
        - What is the potential for the app to grow?
        - What is the potential market size?
        - What’s intentionally left out at the MVP stage?

        8. **Product Roadmap (MVP → V1 → V2 → Full Product)**
        - Describe how the product evolves across 3–4 stages.
        - Mention features, scalability goals, and potential integrations per stage.
        - Include **metrics by phase** (e.g., MVP: early retention; V1: engagement growth; V2: revenue scale).

        9. **User Experience & Design Considerations**
        - **Core user flow:** Step-by-step journey (e.g., onboarding → first transaction → feedback loop).
        - **Onboarding experience:** How users are guided and educated.
        - **Retention or engagement loop:** Notifications, gamification, or habit-forming mechanisms.
        - Include a short **user journey map** that captures key touchpoints and emotional states.

        10. **Technical Implementation Overview**
            - Recommended tech stack (frontend, backend, database, etc.)
            - Architecture choice (monolith, microservices, or serverless)
            - APIs or integrations needed
            - Scalability and performance considerations
            - Deployment environment (e.g., AWS, GCP, Vercel)
            - CI/CD pipeline and monitoring stack (e.g., GitHub Actions, Sentry, Datadog)
            - Include **compliance considerations** such as KYC/AML, GDPR, or data privacy requirements.

        11. **Business Model**
            - Overall business model type (e.g., subscription, SaaS, freemium, transaction-based)
            - Pricing strategy
            - Customer acquisition and retention model
            - Include a **basic monetization forecast** (example: user volume × pricing = projected MRR or ARR)

        12. **Possible Monetization Strategies**
            - List 3–5 monetization options or revenue streams.
            - Include both short-term and long-term opportunities.
            - Examples: freemium plans, in-app purchases, B2B licensing, API usage fees, ads, affiliate marketing, data insights, white-label options, partnerships.
            - Identify which monetization strategies best fit the MVP stage vs. later scaling stages.

        13. **Go-To-Market Strategy**
            - Launch sequence (beta → public launch → expansion)
            - Growth channels (organic, paid, partnerships, influencer, community)
            - Early adopter acquisition tactics
            - Community-building and brand positioning plans
            - Include a brief **content and marketing timeline** (e.g., weekly or monthly themes for launch period)

        14. **Success Metrics**
            - KPIs to track post-launch (user growth, engagement, retention, conversion, churn, and revenue)
            - Break down KPIs per phase (MVP, V1, V2) for clarity.

        15. **Challenges & Risks**
            - Technical, operational, market, or regulatory risks
            - Mitigation strategies and fallback options
            - Include a short “lessons from competitors” insight, if applicable.

        16. **Team & Skills Required**
            - Outline key roles needed for MVP and scaling stages (e.g., Product Manager, Full-Stack Engineer, UI/UX Designer, Security Engineer, Marketing Lead)
            - Briefly state responsibilities per role.

        17. **Long-Term Opportunities**
            - Potential expansions (features, integrations, new markets)
            - Strategic partnerships or ecosystem opportunities
            - Future roadmap for scaling globally
            - End with a **5-Year Vision:** Describe what the app could evolve into — e.g., an intelligent platform, a marketplace, or a full ecosystem.

        Your response should read like a **founder’s detailed product vision document** — 
        clear, strategic, investor-ready, and comprehensive enough for a startup team to use 
        as a step-by-step roadmap from MVP to a fully launched and monetized product."""


# ========== Main Flow ==========

performance_review_template = """You are a strategic product analyst and business reviewer. 
        Given this context:
        {context}

        Generate a **comprehensive performance report** on the product or company, focusing on results, lessons, and future directions.

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

        Your response should read like a **professional product performance report** — 
        data-driven, reflective, and actionable."""

def make_performance_review_chain(llm):
    prompt = PromptTemplate(input_variables=["context"], template=performance_review_template)
    return prompt | llm

def make_idea_chain(llm):
    prompt = PromptTemplate(input_variables=["pain_points"], template=idea_template_single)
    return prompt | llm

def choose_llm(use_json_mode: bool = True):
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
            
            llm = ChatOllama(**llm_kwargs)
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
    )
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

def _invoke_chain_safely(chain, input_data: Dict[str, str], fallback_key: str = "") -> str:
    """
    Safely invoke a chain and extract the response content.
    Handles 403 errors specifically and re-raises other exceptions.
    """
    try:
        result = chain.invoke(input_data)
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
        
        # Test Reddit connection
        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}
        
        thread_text = fetch_reddit_thread(reddit, url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        llm = choose_llm(use_json_mode=False)  # Use text mode for formatted output
        pain_chain = make_painpoint_chain(llm)
        idea_chain = make_idea_chain(llm)
        performance_chain = make_performance_review_chain(llm)

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
                {"context": thread_text}, 
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




def parse_llm_painpoints(raw_text):
    """
    Converts LLM-formatted pain point bullets into a clean JSON structure.
    Handles:
      - Bold titles (e.g., **Title**)
      - Multi-line quotes
      - Implicit quotes
      - Authors following "—"
      - Multiple quotes per bullet section
    Returns:
      - JSON list of pain points with quotes, authors, and number of unique mentions
    """
    

    pain_points = []

    # Split on bullets
    sections = re.split(r"^\s*•", raw_text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    for section in sections:

        # Extract title inside ** **
        title_match = re.search(r"\*\*(.*?)\*\*", section)
        if not title_match:
            continue

        title = title_match.group(1).strip()

        quotes_list = []

        # Check for implicit pain point
        if "Implicit" in section or "implicit" in section:
            quotes_list.append({
                "quote": "(Implicit, not explicitly quoted)",
                "author": None
            })
            num_mentions = 0  # No user explicitly mentioned
        else:
            # Extract quotes (text inside quotes "...")
            raw_quotes = re.findall(r'"(.*?)"', section, flags=re.DOTALL)

            authors_seen = set()
            for q in raw_quotes:
                # Try to extract author for this specific quote
                pattern = re.escape(f'"{q}"') + r'\s*—\s*([^\n]+)'
                author_match = re.search(pattern, section)

                author = author_match.group(1).strip() if author_match else None
                if author:
                    authors_seen.add(author)

                quotes_list.append({
                    "quote": q.strip(),
                    "author": author
                })

            num_mentions = len(authors_seen)

            # If no authors found, count as 1 mention per quote
            if num_mentions == 0:
                num_mentions = len(quotes_list)

        pain_points.append({
            "title": title,
            "quotes": quotes_list,
            "num_mentions": num_mentions
        })

    return pain_points

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
        
        # Test Reddit connection
        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}
        
        thread_text = fetch_reddit_thread(reddit, url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        llm = choose_llm(use_json_mode=False)  # Use text mode for formatted output
        pain_chain = make_painpoint_chain(llm)

        # Invoke chains with error handling
        try:
            pain_resp = _invoke_chain_safely(
                pain_chain, 
                {"thread_text": thread_text}, 
                fallback_key="pain_points"
            )
        except ValueError as e:
            return {"error": str(e)}

        return {
            "pain_points": parse_llm_painpoints(pain_resp),
            "painpoint_count": len(parse_llm_painpoints(pain_resp)),
            "url": url
        }
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in generate_reddit_report_structured: {str(e)}")
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









