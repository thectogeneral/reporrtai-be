import os
from dotenv import load_dotenv
import re

# This will search for a .env file in the current directory or parent directories
load_dotenv()

from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs
import praw
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Prompt for pain point extraction
painpoint_template = """
You are an expert product analyst. Given the following conversation / thread text, extract the main **pain points** that users mention. List each pain point in bullet form, and for each, include a short example quote (1-2 sentences) from the text.

Thread: 
{thread_text}
"""

# Prompt for idea generation
idea_template = """
You are an innovative product strategist and system designer. 
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
as a step-by-step roadmap from MVP to a fully launched and monetized product.
"""


# ========== Main Flow ==========

def make_painpoint_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["thread_text"], template=painpoint_template),
        output_key="pain_points"
    )

def make_idea_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["pain_points"], template=idea_template),
        output_key="app_idea"
    )

def choose_llm():
    """Try Ollama first; if fails, fallback to OpenAI"""
    # You may check some env var to force which you want
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() in ("1", "true", "yes")
    if use_ollama:
        try:
            # You can use ChatOllama (chat-style) or OllamaLLM (text style)
            llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0.7)
            return llm
        except Exception as e:
            print("Ollama init failed, falling back to OpenAI:", e)
    # Fallback to OpenAI
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
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
            return "Error: Reddit API credentials not properly configured"
        
        thread_text = fetch_reddit_thread(reddit, url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            return "Error: Could not fetch thread content or thread is empty"

        llm = choose_llm()
        pain_chain = make_painpoint_chain(llm)
        idea_chain = make_idea_chain(llm)

        # Use invoke (newer) or call rather than .run (deprecated)
        # If using older run API, you may keep run, but better to use invoke
        pain_resp_dict = pain_chain.invoke({"thread_text": thread_text})
        pain_resp = pain_resp_dict["pain_points"]

        idea_resp_dict = idea_chain.invoke({"pain_points": pain_resp})
        idea_resp = idea_resp_dict["app_idea"]

        report = []
        report.append("=== Pain Points ===\n")
        report.append(pain_resp.strip())
        report.append("\n\n=== App Idea & Solution ===\n")
        report.append(idea_resp.strip())
        return "\n".join(report)
        
    except Exception as e:
        # Log the error and return a safe error message
        print(f"Error in generate__reddit_report: {str(e)}")
        return f"Error generating report: {str(e)}"

def save_report(report: str, filename: str = "report.txt"):
    # This will create the file in the **current working directory**
    cwd = os.getcwd()
    fullpath = os.path.join(cwd, filename)
    with open(fullpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {fullpath}")

if __name__ == "__main__":
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
