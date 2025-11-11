import os
from dotenv import load_dotenv
import re

# This will search for a .env file in the current directory or parent directories
load_dotenv()

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import json
import praw
from langchain_classic.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import json-repair, fallback if not available
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("Warning: json-repair not installed. Install with: pip install json-repair")

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
    """Create a chain for pain points extraction using Ollama's format='json'"""
    # Enhanced prompt with clear instructions and examples
    prompt = PromptTemplate(
        template="""You are an expert product analyst. Analyze the conversation/thread text below and extract the main pain points that users are discussing.

        CRITICAL REQUIREMENTS:
        1. Extract REAL, MEANINGFUL pain points from the actual text. Do NOT return empty descriptions.
        2. For EACH pain point, you MUST include a "quote" field with an actual quote from the text (1-2 sentences).
        3. The quote must be copied directly from the thread text, not paraphrased.
        4. Include a "type" field to categorize each pain point (e.g., "Fees", "Performance", "UX", "Security", "Support", "Usability", etc.)
        5. **MANDATORY: You MUST return AT LEAST 3 pain points, preferably 5-7. Do NOT return only 1 or 2 pain points.**

        For each pain point, provide:
        - "type": A category/type for the pain point (e.g., "Fees", "Performance", "UX", "Security", "Support", "Usability")
        - "description": A clear description of the pain point (minimum 20 characters, be specific)
        - "sentiment": The sentiment - must be exactly one of: "negative", "positive", or "neutral"
        - "quote": A DIRECT QUOTE from the text that illustrates this pain point (1-2 sentences, MUST copy actual text from the thread)

        **IMPORTANT: Return AT LEAST 3-7 pain points as a JSON array. NEVER return only 1 pain point. Each pain point MUST have non-empty description, type, AND quote fields.**

Example format:
[
  {{
            "type": "Fees",
            "description": "High transaction fees making small withdrawals unprofitable",
    "sentiment": "negative",
            "quote": "The fees are way too high for small transactions, I'm losing money"
  }},
  {{
            "type": "Performance",
            "description": "Slow withdrawal processing times",
            "sentiment": "negative",
            "quote": "It takes days to get my money, this is frustrating"
  }}
]

Thread text:
{thread_text}

        Now extract the pain points from the above text. For EACH pain point, include a type, description, and a direct quote from the thread text in the "quote" field:""",
            input_variables=["thread_text"]
        )
    
    # Simple chain - Ollama's format="json" will ensure JSON output
    chain = prompt | llm
    return chain

# Prompt for idea generation (multiple ideas with structured data)
idea_template = """
You are an innovative product strategist and system designer. 
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

    CRITICAL INSTRUCTIONS:
    - You MUST respond with ONLY a valid JSON array
    - NO markdown formatting (no ```json```, no **, no *, no headers)
    - NO explanatory text before or after the JSON
    - NO additional commentary
    - Start your response with [ and end with ]
    - Return ONLY the raw JSON array, nothing else

Format your response as a JSON array where each item has:
- "tagline": the catchy tagline
- "problem": what problem it solves
- "product_description": the product description
- "full_detail_report": the comprehensive report

    Example format (return exactly this structure):
    [
    {{
        "tagline": "Fee-free crypto withdrawals",
        "problem": "High fees make small transactions unprofitable",
        "product_description": "A platform that aggregates withdrawal options to find the lowest fees",
        "full_detail_report": "Comprehensive report here..."
    }}
    ]

    Remember: Return ONLY the JSON array starting with [ and ending with ]. No other text.
"""

# Prompt for single idea generation (for backward compatibility)
idea_template_single = """
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

performance_review_template = """
You are a strategic product analyst and business reviewer. 
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
data-driven, reflective, and actionable.
"""

def make_performance_review_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["context"], template=performance_review_template),
        output_key="performance_report"
    )

def make_idea_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["pain_points"], template=idea_template),
        output_key="app_idea"
    )

def convert_markdown_to_json(text: str, llm=None) -> Optional[List[Dict[str, Any]]]:
    """
    Convert markdown-formatted text to JSON using a second LLM call.
    This is a fallback when the LLM doesn't return JSON directly.
    """
    if not llm:
        return None
    
    conversion_prompt = """
    You are a JSON converter. Convert the following markdown text into a valid JSON array.

    The text contains pain points or product ideas. Extract the information and format it as JSON.

    CRITICAL: Return ONLY valid JSON, no markdown, no explanations. Start with [ and end with ].

    Markdown text:
    {markdown_text}

    Return ONLY the JSON array:
    """
    
    try:
        conversion_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["markdown_text"], template=conversion_prompt),
            output_key="json_output"
        )
        result = conversion_chain.invoke({"markdown_text": text})
        json_text = result["json_output"].strip()
        # Try to parse the converted JSON
        parsed = json.loads(json_text)
        return normalize_parsed_json(parsed)
    except Exception as e:
        print(f"Warning: Failed to convert markdown to JSON: {str(e)}")
        return None

def extract_json_from_markdown(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract structured data from markdown text and convert to JSON.
    This handles cases where the LLM returns markdown lists or formatted text.
    """
    try:
        pain_points = []
        lines = text.split('\n')
        current_item = {}
        
        # Skip common intro lines
        skip_patterns = [
            r'^here\s+is',
            r'^here\s+are',
            r'^the\s+main',
            r'^key\s+points',
            r'^summary',
            r'^topic:',
            r'^overview'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line might signal end of current item
                if current_item:
                    pain_points.append(current_item)
                    current_item = {}
                continue
            
            # Skip headers and intro lines
            if line.startswith('#') or any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                continue
            
            # Remove markdown formatting
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            clean_line = re.sub(r'\*([^*]+)\*', r'\1', clean_line)
            clean_line = re.sub(r'^[-•]\s*', '', clean_line)
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)  # Remove numbered lists
            
            # Look for key-value patterns (Description:, Sentiment:, Quote:, etc.)
            if ':' in clean_line:
                parts = clean_line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    if any(word in key for word in ['description', 'pain point', 'issue', 'problem', 'concern']):
                        if current_item:
                            pain_points.append(current_item)
                        current_item = {"description": value[:300], "sentiment": "neutral", "quote": ""}
                    elif 'sentiment' in key:
                        if current_item:
                            # Normalize sentiment
                            sentiment_val = value.lower().strip()
                            if 'negative' in sentiment_val or 'bad' in sentiment_val or 'poor' in sentiment_val:
                                current_item["sentiment"] = "negative"
                            elif 'positive' in sentiment_val or 'good' in sentiment_val or 'great' in sentiment_val:
                                current_item["sentiment"] = "positive"
                            else:
                                current_item["sentiment"] = "neutral"
                    elif 'quote' in key or 'example' in key:
                        if current_item:
                            current_item["quote"] = value.strip('"').strip("'")[:200]
            
            # Look for bullet points or list items
            elif (clean_line.startswith('-') or 
                  clean_line.startswith('•') or 
                  clean_line.startswith('*') or
                  re.match(r'^\d+[\.\)]', clean_line)):
                content = re.sub(r'^[-•*]\s*', '', clean_line)
                content = re.sub(r'^\d+[\.\)]\s*', '', content)
                if content and len(content) > 15:
                    if current_item:
                        pain_points.append(current_item)
                    current_item = {"description": content[:300], "sentiment": "neutral", "quote": ""}
            
            # Look for standalone sentences that might be pain points
            elif len(clean_line) > 20 and len(clean_line) < 300:
                # Check if it looks like a pain point (contains keywords)
                pain_keywords = ['fee', 'cost', 'expensive', 'slow', 'difficult', 'problem', 'issue', 
                                'frustrating', 'annoying', 'wish', 'need', 'want', 'should', 'better']
                if any(keyword in clean_line.lower() for keyword in pain_keywords):
                    if current_item:
                        pain_points.append(current_item)
                    current_item = {"description": clean_line[:300], "sentiment": "neutral", "quote": ""}
        
        # Add the last item
        if current_item:
            pain_points.append(current_item)
        
        if pain_points:
            return pain_points
    except Exception as e:
        print(f"Warning: Failed to extract JSON from markdown: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    return None

def normalize_parsed_json(parsed: Any) -> List[Dict[str, Any]]:
    """
    Normalize parsed JSON to ensure it's always a list of dictionaries.
    Handles cases where parsed might be a list containing lists or mixed types.
    """
    if isinstance(parsed, list):
        normalized = []
        for item in parsed:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, list):
                # Flatten nested lists
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        normalized.append(sub_item)
        return normalized if normalized else []
    elif isinstance(parsed, dict):
        return [parsed]
    else:
        return []

def parse_json_response(text: str, llm=None) -> List[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks and extra text"""
    if not text or not text.strip():
        return []
    
    # Store original for error messages
    original_text = text
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Strategy 1: Simple regex extraction (try this first - it's fast and works for most cases)
    # Try multiple regex patterns, from simple to more complex
    regex_patterns = [
        r'\[[^\]]*\{[^}]*\}[^\]]*\]',  # Array with objects
        r'\[.*?\]',  # Simple array (non-greedy)
        r'\[.*\]',   # Simple array (greedy)
    ]
    
    for pattern in regex_patterns:
        json_array_match = re.search(pattern, text, re.DOTALL)
        if json_array_match:
            try:
                json_text = json_array_match.group(0)
                # Clean up common issues before parsing
                json_text = re.sub(r',\s*\]', ']', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*}', '}', json_text)
                parsed = json.loads(json_text)
                result = normalize_parsed_json(parsed)
                if result:
                    print(f"Successfully extracted JSON using regex pattern: {pattern}")
                    return result
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            json_text = code_block_match.group(1).strip()
            parsed = json.loads(json_text)
            result = normalize_parsed_json(parsed)
            if result:
                print("Successfully extracted JSON from code block")
                return result
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: More careful extraction with balanced brackets (for nested/complex JSON)
    # Find JSON array by looking for balanced brackets
    start_idx = text.find('[')
    if start_idx != -1:
        bracket_count = 0
        brace_count = 0
        in_string = False
        escape_next = False
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '[':
                                bracket_count += 1
                elif char == ']':
                                bracket_count -= 1
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                
                # Found complete array when brackets and braces are balanced
                if bracket_count == 0 and brace_count == 0 and i > start_idx:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            text = text[start_idx:end_idx]
    
    # If we still don't have a valid array structure, try regex fallback
    if not text.startswith('['):
        json_match = re.search(r'(\[[\s\S]*\])', text)
    if json_match:
        text = json_match.group(1).strip()
    
    # Try to clean up common issues
    # Remove any text before the first [
    if '[' in text:
        text = text[text.index('['):]
    # Remove any text after the last ]
    if ']' in text:
        text = text[:text.rindex(']') + 1]
    
    # Try to fix common JSON issues
    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r',\s*}', '}', text)
    # Remove any markdown formatting that might have leaked in
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
    
    # Check if JSON appears incomplete (ends with incomplete string or key)
    is_incomplete = False
    if text:
        # Check for incomplete strings (ends with unclosed quote)
        if text.count('"') % 2 != 0:
            is_incomplete = True
        # Check for incomplete keys (ends with "key": or "key": "value)
        if re.search(r'"[^"]*":\s*"[^"]*$', text):
            is_incomplete = True
        # Check if ends with incomplete structure
        if text.rstrip().endswith(('"', ':', ',')):
            is_incomplete = True
    
    try:
        parsed = json.loads(text)
        return normalize_parsed_json(parsed)
    except json.JSONDecodeError as e:
        # Try to fix incomplete JSON by closing open structures
        if is_incomplete:
            try:
                # Try to complete the JSON structure
                # Count open braces and brackets
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
                
                # Close incomplete string if needed
                if text.count('"') % 2 != 0:
                    # Find the last unclosed quote and close it
                    last_quote_idx = text.rfind('"')
                    if last_quote_idx != -1:
                        # Check if it's a key or value
                        after_quote = text[last_quote_idx+1:].strip()
                        if after_quote.startswith(':'):
                            # It's a key, add empty string value
                            text = text[:last_quote_idx+1] + '": ""'
                        else:
                            # It's a value, just close it
                            text = text[:last_quote_idx+1] + '"'
                
                # Close open braces
                text += '}' * open_braces
                # Close open brackets
                text += ']' * open_brackets
                
                # Remove trailing commas
                text = re.sub(r',\s*([}\]])', r'\1', text)
                
                # Try parsing again
                parsed = json.loads(text)
                print("Warning: JSON was incomplete but fixed. Consider increasing max_tokens.")
                return normalize_parsed_json(parsed)
            except Exception:
                pass
        
        # Last attempt: try to find the largest valid JSON structure
        try:
            # Look for the first complete JSON array
            start_idx = text.find('[')
            if start_idx != -1:
                # Try to find matching closing bracket with proper brace counting
                bracket_count = 0
                brace_count = 0
                in_string = False
                escape_next = False
                end_idx = start_idx
                
                for i in range(start_idx, len(text)):
                    char = text[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                        elif char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                        
                        if bracket_count == 0 and brace_count == 0 and i > start_idx:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_text = text[start_idx:end_idx]
                    # Clean up the extracted text
                    json_text = re.sub(r',\s*\]', ']', json_text)
                    json_text = re.sub(r',\s*}', '}', json_text)
                    parsed = json.loads(json_text)
                    return normalize_parsed_json(parsed)
        except Exception as inner_e:
            pass
        
        # Strategy 4: Try json-repair library if available
        if HAS_JSON_REPAIR:
            try:
                repaired_json = json_repair.repair_json(text)
                parsed = json.loads(repaired_json)
                print("Successfully repaired JSON using json-repair library")
                return normalize_parsed_json(parsed)
            except Exception as repair_error:
                print(f"json-repair failed: {str(repair_error)}")
        
        # Strategy 5: Try to extract structured data from markdown
        markdown_result = extract_json_from_markdown(original_text)
        if markdown_result:
            print("Successfully extracted JSON from markdown structure")
            return markdown_result
        
        # Strategy 6: Use a second LLM call to convert markdown to JSON
        if llm:
            conversion_result = convert_markdown_to_json(original_text, llm)
            if conversion_result:
                print("Successfully converted markdown to JSON using LLM")
                return conversion_result
        
        # If all parsing attempts fail, log the full response for debugging
        print(f"Warning: Failed to parse JSON response after all strategies. Error: {str(e)}")
        print(f"Response length: {len(original_text)} characters")
        print(f"Extracted text length: {len(text)} characters")
        print(f"First 500 chars: {original_text[:500]}")
        if len(original_text) > 500:
            print(f"Last 200 chars: {original_text[-200:]}")
        if is_incomplete:
            print("Note: JSON appears to be incomplete (truncated). Consider increasing LLM_MAX_TOKENS.")
        return []

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
            # Use ChatOllama with format="json" to force JSON output
            llm_kwargs = {
                "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "temperature": temperature,
                "num_predict": max_tokens
            }
            
            # Add format="json" if use_json_mode is True
            if use_json_mode:
                llm_kwargs["format"] = "json"
                print("DEBUG: Using Ollama with format='json'")
            
            llm = ChatOllama(**llm_kwargs)
            return llm
        except Exception as e:
            print("Ollama init failed, falling back to OpenAI:", e)
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
    report.append("===================================== Pain Points =====================================\n")
    report.append(result["pain_points"].strip())
    report.append("\n\n===================================== App Idea & Solution =====================================\n")
    report.append(result["app_idea"].strip())
    report.append("\n\n===================================== Performance Review =====================================\n")
    report.append(result["performance_review"].strip())
    return "\n".join(report)

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

        llm = choose_llm()
        pain_chain = make_painpoint_chain(llm)
        idea_chain = make_idea_chain(llm)
        performance_chain = make_performance_review_chain(llm)

        # Use invoke (newer) or call rather than .run (deprecated)
        # If using older run API, you may keep run, but better to use invoke
        pain_resp_dict = pain_chain.invoke({"thread_text": thread_text})
        pain_resp = pain_resp_dict["pain_points"]

        idea_resp_dict = idea_chain.invoke({"pain_points": pain_resp})
        idea_resp = idea_resp_dict["app_idea"]

        performance_resp_dict = performance_chain.invoke({"context": thread_text})
        performance_resp = performance_resp_dict["performance_report"]

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

def extract_quote_from_text(description: str, thread_text: str) -> str:
    """
    Extract a relevant quote from the thread text based on the pain point description.
    This is a fallback when the LLM doesn't provide quotes.
    """
    if not description or not thread_text:
        print(f"DEBUG extract_quote: Missing description or thread_text. desc={bool(description)}, thread={bool(thread_text)}")
        return ""
    
    # Extract keywords from description
    keywords = []
    # Remove common words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'not', 'has', 'have', 'this', 'that', 'these', 'those'}
    words = description.lower().split()
    # Extract keywords - include words >= 3 chars, and also important short words like "fee", "usd", "usdt"
    important_short_words = {'fee', 'fees', 'usd', 'usdt', 'btc', 'eth', 'crypto', 'coinbase', 'kraken', 'transfer', 'withdraw'}
    keywords = [w.strip('.,!?;:') for w in words if (w not in stop_words and len(w) >= 3) or w in important_short_words]
    
    print(f"DEBUG extract_quote: Extracted keywords from '{description[:50]}...': {keywords[:8]}")
    
    if not keywords:
        print("DEBUG extract_quote: No keywords extracted")
        return ""
    
    # Split thread into sentences - use more flexible splitting
    # First try standard sentence splitting
    sentences = re.split(r'[.!?]\s+', thread_text)
    # Also split on newlines that might indicate separate thoughts
    all_sentences = []
    for sent in sentences:
        # Further split on newlines
        sub_sents = sent.split('\n')
        for sub_sent in sub_sents:
            sub_sent = sub_sent.strip()
            if sub_sent:
                all_sentences.append(sub_sent)
    
    print(f"DEBUG extract_quote: Split into {len(all_sentences)} sentences")
    
    # Find sentences that contain keywords from the description
    best_match = ""
    best_score = 0
    
    for sentence in all_sentences:
        sentence = sentence.strip()
        # More lenient length requirements
        if len(sentence) < 15 or len(sentence) > 400:
            continue
        
        # Count how many keywords appear in this sentence (case-insensitive)
        sentence_lower = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        # Prefer sentences with multiple keywords and reasonable length
        if score > best_score and score > 0:
            best_score = score
            best_match = sentence
    
    # If we found a good match, return it
    if best_match:
        print(f"DEBUG extract_quote: Found match with score {best_score}: {best_match[:150]}...")
        return best_match[:250]  # Limit quote length
    
    # Fallback: return first sentence that contains any keyword (even partial matches)
    for sentence in all_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 15 and len(sentence) <= 300:
            sentence_lower = sentence.lower()
            # Check if any keyword appears in the sentence
            if any(keyword in sentence_lower for keyword in keywords[:5]):  # Check top 5 keywords
                print(f"DEBUG extract_quote: Found fallback match: {sentence[:150]}...")
                return sentence[:250]
    
    # Last resort: try to find sentences with partial keyword matches
    for sentence in all_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 20 and len(sentence) <= 250:
            sentence_lower = sentence.lower()
            # Check if any significant part of keywords match
            for keyword in keywords[:3]:
                if len(keyword) >= 4 and keyword[:4] in sentence_lower:
                    print(f"DEBUG extract_quote: Found partial match for '{keyword}': {sentence[:150]}...")
                    return sentence[:250]
    
    print(f"DEBUG extract_quote: No match found for description: {description[:50]}")
    print(f"DEBUG extract_quote: Keywords searched: {keywords[:5]}")
    print(f"DEBUG extract_quote: First 200 chars of thread: {thread_text[:200]}")
    return ""

def count_commenters_for_pain_point(pain_point_description: str, pain_point_quote: str, thread_text: str) -> int:
    """
    Count how many unique commenters/users mentioned this pain point.
    Looks for comments that contain keywords from the pain point description or quote.
    """
    if not thread_text or not pain_point_description:
        return 0
    
    # Extract keywords from description and quote
    keywords = []
    text_to_analyze = f"{pain_point_description} {pain_point_quote}".lower()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'not', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    words = text_to_analyze.split()
    important_short_words = {'fee', 'fees', 'usd', 'usdt', 'btc', 'eth', 'crypto', 'coinbase', 'kraken', 'transfer', 'withdraw', 'slow', 'fast', 'bug', 'error'}
    
    keywords = [w.strip('.,!?;:') for w in words if (w not in stop_words and len(w) >= 3) or w in important_short_words]
    
    if not keywords:
        return 0
    
    # Extract unique commenters from thread text
    # Pattern: "Comment by {author}: {body}"
    comment_pattern = r'Comment by ([^:]+):\s*(.+?)(?=\nComment by|\nTitle:|$)'
    matches = re.findall(comment_pattern, thread_text, re.MULTILINE | re.DOTALL)
    
    unique_commenters = set()
    
    for author, comment_body in matches:
        comment_lower = comment_body.lower()
        # Check if this comment contains any keywords from the pain point
        if any(keyword in comment_lower for keyword in keywords[:10]):  # Check top 10 keywords
            # Also check if the quote appears in this comment (if quote is provided)
            if pain_point_quote:
                quote_words = pain_point_quote.lower().split()
                quote_keywords = [w.strip('.,!?;:') for w in quote_words if len(w) >= 4]  # Use longer words from quote
                if quote_keywords and any(kw in comment_lower for kw in quote_keywords[:5]):
                    unique_commenters.add(author.strip())
            else:
                unique_commenters.add(author.strip())
    
    # Also check the title/post itself (count as 1 if it matches)
    title_match = re.search(r'Title:\s*(.+?)(?=\nComment by|\nTitle:|$)', thread_text, re.MULTILINE | re.DOTALL)
    if title_match:
        title_text = title_match.group(1).lower()
        if any(keyword in title_text for keyword in keywords[:10]):
            unique_commenters.add("OP")  # Original Poster
    
    return len(unique_commenters)

def extract_pain_points_fallback(thread_text: str) -> List[Dict[str, Any]]:
    """
    Fallback function to extract pain points directly from thread text using keyword matching.
    This is used when the LLM fails to return valid JSON.
    """
    if not thread_text or len(thread_text.strip()) < 20:
        return []
    
    pain_points = []
    
    # Pain point keywords and their associated types
    pain_patterns = {
        "Fees": ["fee", "fees", "cost", "expensive", "charge", "pricing", "price", "costly", "overpriced"],
        "Performance": ["slow", "lag", "delay", "timeout", "freeze", "crash", "bug", "glitch", "broken"],
        "UX": ["confusing", "complicated", "difficult", "hard to use", "unclear", "unintuitive", "clunky"],
        "Support": ["support", "help", "customer service", "response", "waiting", "no reply"],
        "Security": ["security", "hack", "breach", "vulnerability", "unsafe", "risk", "scam"],
        "Usability": ["can't", "cannot", "unable", "impossible", "doesn't work", "not working", "broken"],
        "Limitations": ["limit", "restriction", "can't", "cannot", "not allowed", "blocked"],
    }
    
    # Split thread into sentences
    sentences = re.split(r'[.!?]\s+', thread_text)
    all_sentences = []
    for sent in sentences:
        sub_sents = sent.split('\n')
        for sub_sent in sub_sents:
            sub_sent = sub_sent.strip()
            if sub_sent and len(sub_sent) > 15:
                all_sentences.append(sub_sent)
    
    # Track which sentences we've already used
    used_sentences = set()
    
    # Look for pain points by matching keywords
    for pain_type, keywords in pain_patterns.items():
        for sentence in all_sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains any keyword for this pain type
            if any(keyword in sentence_lower for keyword in keywords):
                # Avoid duplicates
                sentence_hash = hash(sentence[:100])
                if sentence_hash not in used_sentences:
                    used_sentences.add(sentence_hash)
                    
                    # Determine sentiment
                    sentiment = "neutral"
                    negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "frustrating", "annoying"]
                    positive_words = ["good", "great", "excellent", "love", "amazing", "perfect"]
                    
                    if any(word in sentence_lower for word in negative_words):
                        sentiment = "negative"
                    elif any(word in sentence_lower for word in positive_words):
                        sentiment = "positive"
                    
                    # Create a description from the sentence (first 200 chars)
                    description = sentence[:200].strip()
                    if len(description) > 20:  # Only add if meaningful
                        # Count commenters for this pain point
                        commenters_count = count_commenters_for_pain_point(description, sentence[:250], thread_text)
                        
                        pain_points.append({
                            "description": description,
                            "sentiment": sentiment,
                            "quote": sentence[:250],  # Use the sentence as quote
                            "type": pain_type,
                            "commenters_count": commenters_count
                        })
                    
                    # Limit to avoid too many pain points
                    if len(pain_points) >= 7:
                        break
        
        if len(pain_points) >= 7:
            break
    
    # Return pain points even if less than 3 (better than nothing)
    if pain_points:
        return pain_points[:7]  # Return up to 7
    
    return []

def generate_comprehensive_report(url: str) -> Dict[str, Any]:
    """
    Extract pain points from the discussion/thread.
    Returns pain points with sentiment labels and the URL.
    """
    try:
        # Initialize Reddit client
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "app-idea-generator"
        )
        
        if not reddit.read_only:
            return {"error": "Error: Reddit API credentials not properly configured"}
        
        thread_text = fetch_reddit_thread(reddit, url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            return {"error": "Error: Could not fetch thread content or thread is empty"}

        llm = choose_llm()
        pain_chain = make_painpoint_chain(llm)

        # Extract pain points using StructuredOutputParser
        max_retries = int(os.getenv("MAX_RETRIES", "3"))  # Increased from 2 to 3
        pain_points_list = []
        
        # Always call the LLM at least once
        for attempt in range(max(max_retries, 1)):
            try:
                print(f"DEBUG: Calling LLM with format='json' (attempt {attempt + 1})...")
                # The chain returns a message object with content (JSON string)
                result = pain_chain.invoke({"thread_text": thread_text})
                
                print(f"DEBUG: Result type: {type(result)}")
                
                # Extract JSON string from result
                json_text = None
                if hasattr(result, 'content'):
                    json_text = result.content
                    print(f"DEBUG: Got content from message object, length: {len(json_text)}")
                elif isinstance(result, str):
                    json_text = result
                    print(f"DEBUG: Got string result, length: {len(json_text)}")
                elif isinstance(result, dict):
                    # If it's already parsed
                    if "pain_points" in result:
                        pain_points_list = []
                        for pp in result["pain_points"]:
                            # Handle different field names
                            description = (
                                pp.get("description", "") or 
                                pp.get("pain_point", "") or 
                                pp.get("issue", "") or
                                pp.get("problem", "") or
                                str(pp.get("type", ""))
                            ).strip()
                            
                            if description and len(description) > 10:
                                pain_type = pp.get("type", "").strip() or None
                                pain_point_dict = {
                                    "description": description,
                                    "sentiment": pp.get("sentiment", "neutral"),
                                    "quote": pp.get("quote", "").strip()
                                }
                                if pain_type:
                                    pain_point_dict["type"] = pain_type
                                pain_points_list.append(pain_point_dict)
                        print(f"DEBUG: Extracted {len(pain_points_list)} pain points from dict")
                        if pain_points_list:
                            break
                    else:
                        json_text = json.dumps(result)
                else:
                    json_text = str(result)
                
                if json_text:
                    print(f"DEBUG: JSON text preview: {json_text[:500]}...")
                    print(f"DEBUG: Full JSON text: {json_text}")
                    # Parse the JSON response
                    try:
                        # Try direct JSON parsing first (Ollama format="json" should return valid JSON)
                        parsed = json.loads(json_text.strip())
                        print(f"DEBUG: Successfully parsed JSON directly. Type: {type(parsed)}")
                        print(f"DEBUG: Parsed content: {parsed}")
                        
                        # Handle different JSON structures
                        if isinstance(parsed, list):
                            print(f"DEBUG: Parsed is a list with {len(parsed)} items")
                            # If it's a list of pain points - filter out empty ones
                            pain_points_list = []
                            for idx, pp in enumerate(parsed):
                                print(f"DEBUG: Processing list item {idx}: {pp}")
                                if not isinstance(pp, dict):
                                    print(f"DEBUG: Item {idx} is not a dict, skipping")
                                    continue
                                    
                                # Handle different field names
                                description = (
                                    pp.get("description", "") or 
                                    pp.get("pain_point", "") or 
                                    pp.get("issue", "") or
                                    pp.get("problem", "") or
                                    str(pp.get("type", ""))
                                ).strip()
                                
                                print(f"DEBUG: Item {idx} extracted description: '{description}', length: {len(description)}")
                                
                                quote = (
                                    pp.get("quote", "") or 
                                    pp.get("example", "") or
                                    pp.get("text", "") or
                                    pp.get("excerpt", "")
                                ).strip()
                                
                                # If quote is still empty, try to extract it now
                                if not quote:
                                    print(f"DEBUG: Quote empty in parsed JSON for '{description[:50]}...', will extract later")
                                
                                sentiment = pp.get("sentiment", "neutral")
                                # Validate sentiment
                                if sentiment not in ["negative", "positive", "neutral"]:
                                    sentiment = "neutral"
                                
                                # Only include pain points with meaningful content
                                if description and len(description) > 10:
                                    # Extract type field if present
                                    pain_type = pp.get("type", "").strip() or None
                                    
                                    pain_point_dict = {
                                        "description": description,
                                        "sentiment": sentiment,
                                        "quote": quote if quote else ""  # Keep empty string for now, will fill later
                                    }
                                    # Add type if present
                                    if pain_type:
                                        pain_point_dict["type"] = pain_type
                                    pain_points_list.append(pain_point_dict)
                                else:
                                    print(f"DEBUG: Skipping empty pain point: {pp}")
                        elif isinstance(parsed, dict):
                            if "pain_points" in parsed:
                                # If it's wrapped in an object - filter out empty ones
                                pain_points_list = []
                                for pp in parsed["pain_points"]:
                                    # Handle different field names
                                    description = (
                                        pp.get("description", "") or 
                                        pp.get("pain_point", "") or 
                                        pp.get("issue", "") or
                                        pp.get("problem", "") or
                                        str(pp.get("type", ""))
                                    ).strip()
                                    
                                    quote = (
                                        pp.get("quote", "") or 
                                        pp.get("example", "") or
                                        pp.get("text", "")
                                    ).strip()
                                    
                                    sentiment = pp.get("sentiment", "neutral")
                                    if sentiment not in ["negative", "positive", "neutral"]:
                                        sentiment = "neutral"
                                    
                                    if description and len(description) > 10:
                                        # Extract type field if present
                                        pain_type = pp.get("type", "").strip() or None
                                        
                                        pain_point_dict = {
                                            "description": description,
                                            "sentiment": sentiment,
                                            "quote": quote if quote else ""
                                        }
                                        # Add type if present
                                        if pain_type:
                                            pain_point_dict["type"] = pain_type
                                        pain_points_list.append(pain_point_dict)
                            else:
                                # Single pain point object - try to extract description from various fields
                                description = (
                                    parsed.get("description", "") or 
                                    parsed.get("pain_point", "") or 
                                    parsed.get("issue", "") or
                                    parsed.get("problem", "") or
                                    str(parsed.get("type", ""))
                                ).strip()
                                
                                if description and len(description) > 10:
                                    sentiment = parsed.get("sentiment", "neutral")
                                    if sentiment not in ["negative", "positive", "neutral"]:
                                        sentiment = "neutral"
                                    
                                    # Extract type field if present
                                    pain_type = parsed.get("type", "").strip() or None
                                    
                                    pain_point_dict = {
                                        "description": description,
                                        "sentiment": sentiment,
                                        "quote": parsed.get("quote", "").strip()
                                    }
                                    # Add type if present
                                    if pain_type:
                                        pain_point_dict["type"] = pain_type
                                    pain_points_list = [pain_point_dict]
                        
                        if pain_points_list:
                            print(f"DEBUG: Successfully extracted {len(pain_points_list)} pain points from JSON")
                            # Validate that we actually have descriptions
                            valid_count = sum(1 for pp in pain_points_list if isinstance(pp, dict) and pp.get("description", "").strip() and len(pp.get("description", "").strip()) > 10)
                            print(f"DEBUG: {valid_count} out of {len(pain_points_list)} pain points have valid descriptions")
                            
                            # Check if we have enough pain points (at least 3)
                            if valid_count >= 3:
                                print(f"DEBUG: Got {valid_count} valid pain points, proceeding")
                                break
                            elif valid_count > 0:
                                print(f"DEBUG: Only got {valid_count} valid pain points (need at least 3), will retry")
                                pain_points_list = []
                            else:
                                print(f"DEBUG: All {len(pain_points_list)} pain points have empty descriptions, will retry")
                                pain_points_list = []
                    except json.JSONDecodeError as json_err:
                        print(f"DEBUG: Direct JSON parsing failed: {json_err}")
                        # Fall back to parse_json_response for extraction
                        pain_points_list = parse_json_response(json_text, llm=llm)
                        if pain_points_list:
                            print(f"DEBUG: Parsed {len(pain_points_list)} pain points using parse_json_response")
                            break
                
                if attempt < max_retries:
                    print(f"Warning: Failed to extract pain points on attempt {attempt + 1}. Retrying...")
                else:
                    print(f"Warning: Failed after {attempt + 1} attempts.")
            except Exception as e:
                print(f"Error calling LLM on attempt {attempt + 1}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                if attempt >= max_retries:
                    # On final attempt, try fallback to old method
                    print("DEBUG: Falling back to manual parsing...")
                    try:
                        # Create a simple chain without parser as fallback
                        fallback_prompt = PromptTemplate(
                            input_variables=["thread_text"],
                            template="""Extract the main pain points from this discussion. Return as JSON array with description, sentiment, and quote fields. 
                            Thread: {thread_text}
                            Return JSON only:"""
                        )
                        fallback_chain = fallback_prompt | llm
                        fallback_result = fallback_chain.invoke({"thread_text": thread_text})
                        if hasattr(fallback_result, 'content'):
                            fallback_text = fallback_result.content
                        else:
                            fallback_text = str(fallback_result)
                        pain_points_list = parse_json_response(fallback_text, llm=llm)
                        if pain_points_list:
                            break
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {str(fallback_error)}")
                        raise
        
        # If no pain points extracted, try aggressive fallback extraction
        if not pain_points_list:
            print("DEBUG: No pain points extracted from LLM, trying fallback extraction from thread text")
            # Try to extract pain points directly from thread text using keyword matching
            fallback_pain_points = extract_pain_points_fallback(thread_text)
            if fallback_pain_points:
                print(f"DEBUG: Fallback extraction found {len(fallback_pain_points)} pain points")
                pain_points_list = fallback_pain_points
            else:
                print("DEBUG: Fallback extraction also failed, creating generic fallback")
            pain_points_list = [{
                    "description": "Unable to extract pain points from the discussion",
                "sentiment": "neutral",
                "quote": ""
            }]

        # Filter out empty pain points
        filtered_pain_points = []
        for idx, item in enumerate(pain_points_list):
            # Skip non-dict items (strings, lists, etc.)
            if not isinstance(item, dict):
                print(f"DEBUG: Skipping non-dict item at index {idx}, type: {type(item)}, value: {item}")
                continue
                
            # Handle different field names
            description = (
                item.get("description", "") or 
                item.get("pain_point", "") or 
                item.get("issue", "") or
                item.get("problem", "") or
                str(item.get("type", ""))
            ).strip()
            
            # Only keep pain points with meaningful descriptions
            if description and len(description) > 10:
                quote = item.get("quote", "").strip()
                # If quote is missing, try to extract it from thread text
                if not quote:
                    print(f"DEBUG: Missing quote for pain point '{description[:50]}...', attempting to extract from thread text")
                    quote = extract_quote_from_text(description, thread_text)
                    if quote:
                        print(f"DEBUG: Extracted quote: {quote[:100]}...")
                
                # Extract type field if present
                pain_type = item.get("type", "").strip() or None
                
                # Count commenters who mentioned this pain point
                commenters_count = count_commenters_for_pain_point(description, quote, thread_text)
                
                pain_point_dict = {
                    "description": description,
                    "sentiment": item.get("sentiment", "neutral"),
                    "quote": quote
                }
                # Add type if present
                if pain_type:
                    pain_point_dict["type"] = pain_type
                # Add commenters count
                pain_point_dict["commenters_count"] = commenters_count
                filtered_pain_points.append(pain_point_dict)
            else:
                print(f"DEBUG: Filtered out empty pain point at index {idx}: {item}")
                print(f"DEBUG: Description value: '{description}', length: {len(description)}")
        
        pain_points_list = filtered_pain_points
        print(f"DEBUG: After filtering, have {len(pain_points_list)} pain points")
        
        # If no pain points extracted or all were empty, create fallback
        if not pain_points_list:
            print("DEBUG: No valid pain points extracted (all were empty), creating fallback")
            pain_points_list = [{
                "description": "Unable to extract pain points from the discussion",
                "sentiment": "neutral",
                "quote": ""
            }]
        
        # Final validation: ensure pain_points_list is a list of dictionaries
        final_pain_points = []
        print(f"DEBUG: Final validation - pain_points_list has {len(pain_points_list)} items")
        for idx, item in enumerate(pain_points_list):
            print(f"DEBUG: Validating item {idx}, type: {type(item)}, full item: {item}")
            if isinstance(item, dict):
                # Handle different field names in final validation too
                description = (
                    item.get("description", "") or 
                    item.get("pain_point", "") or 
                    item.get("issue", "") or
                    item.get("problem", "") or
                    str(item.get("type", ""))
                ).strip()
                
                print(f"DEBUG: Item {idx} description: '{description}', length: {len(description)}")
                
                # Skip empty descriptions
                if not description or len(description) < 10:
                    print(f"DEBUG: Skipping item {idx} with empty/short description")
                    continue
                # Skip the fallback message if we have other items
                if description == "Unable to extract pain points from the discussion" and len(pain_points_list) > 1:
                    print(f"DEBUG: Skipping fallback message at index {idx}")
                    continue
                # Ensure we have the correct structure
                quote = item.get("quote", "").strip()
                # If quote is still missing, try to extract it from thread text
                if not quote:
                    print(f"DEBUG: Missing quote in final validation for '{description[:50]}...', extracting from thread")
                    quote = extract_quote_from_text(description, thread_text)
                
                # Extract type field if present
                pain_type = item.get("type", "").strip() or None
                
                # Count commenters who mentioned this pain point
                commenters_count = count_commenters_for_pain_point(description, quote, thread_text)
                print(f"DEBUG: Pain point '{description[:50]}...' mentioned by {commenters_count} commenters")
                
                pain_point_dict = {
                    "description": description,
                    "sentiment": item.get("sentiment", "neutral"),
                    "quote": quote
                }
                # Add type if present
                if pain_type:
                    pain_point_dict["type"] = pain_type
                # Add commenters count
                pain_point_dict["commenters_count"] = commenters_count
                final_pain_points.append(pain_point_dict)
            elif isinstance(item, list):
                # Flatten nested lists
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        desc = sub_item.get("description", "").strip()
                        if desc and len(desc) > 10:
                            final_pain_points.append(sub_item)
        
        # If we still don't have valid pain points, create a fallback
        if not final_pain_points:
            print("DEBUG: No valid pain points found after validation, creating fallback")
            final_pain_points = [{
                "description": "Unable to extract pain points from the discussion",
                "sentiment": "neutral",
                "quote": ""
            }]
        
        print(f"DEBUG: Returning {len(final_pain_points)} pain points")
        return {
            "pain_points": final_pain_points,
            "url": url
        }
        
    except Exception as e:
        print(f"Error in generate_comprehensive_report: {str(e)}")
        return {"error": f"Error generating comprehensive report: {str(e)}"}

def save_report(report: str, filename: str = "report.txt"):
    # This will create the file in the **current working directory**
    cwd = os.getcwd()
    fullpath = os.path.join(cwd, filename)
    with open(fullpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {fullpath}")

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

# Request/Response models
class ReportRequest(BaseModel):
    url: str

class ReportResponse(BaseModel):
    pain_points: str
    app_idea: str
    performance_review: str
    url: str

class ErrorResponse(BaseModel):
    error: str

class PainPointsResponse(BaseModel):
    pain_points: str
    url: str

class PainPoint(BaseModel):
    description: str
    sentiment: str  # "negative", "positive", or "neutral"
    quote: str
    type: Optional[str] = None  # Category/type of the pain point (e.g., "Fees", "Performance", "UX")
    commenters_count: int = 0  # Number of unique commenters/users who mentioned this pain point

class ComprehensiveReportResponse(BaseModel):
    pain_points: List[PainPoint]
    pain_points_count: int  # Number of pain points
    url: str

@app.get("/")
async def root():
    return {"message": "Reporrt AI API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a full report (pain points, app idea, and performance review) from a Reddit URL.
    """
    try:
        # Validate URL
        reddit_pattern = r'https?://(?:www\.)?reddit\.com/r/.*?/comments/.*'
        if not re.match(reddit_pattern, request.url):
            raise HTTPException(status_code=400, detail="Invalid Reddit URL format")
        
        result = generate_reddit_report_structured(request.url)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ReportResponse(
            pain_points=result["pain_points"],
            app_idea=result["app_idea"],
            performance_review=result["performance_review"],
            url=result["url"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/api/painpoints", response_model=PainPointsResponse)
async def get_pain_points(request: ReportRequest):
    """
    Extract only pain points from a Reddit thread URL.
    """
    try:
        # Validate URL
        reddit_pattern = r'https?://(?:www\.)?reddit\.com/r/.*?/comments/.*'
        if not re.match(reddit_pattern, request.url):
            raise HTTPException(status_code=400, detail="Invalid Reddit URL format")
        
        # Initialize Reddit client
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT") or "app-idea-generator"
        )
        
        if not reddit.read_only:
            raise HTTPException(status_code=500, detail="Reddit API credentials not properly configured")
        
        thread_text = fetch_reddit_thread(reddit, request.url)
        
        if not thread_text or len(thread_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not fetch thread content or thread is empty")
        
        llm = choose_llm()
        pain_chain = make_painpoint_chain(llm)
        
        pain_resp_dict = pain_chain.invoke({"thread_text": thread_text})
        pain_resp = pain_resp_dict["pain_points"]
        
        return PainPointsResponse(
            pain_points=pain_resp.strip(),
            url=request.url
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting pain points: {str(e)}")

@app.post("/api/comprehensive", response_model=ComprehensiveReportResponse)
async def get_comprehensive_report(request: ReportRequest):
    """
    Extract pain points from the discussion/thread.
    Returns a list of pain points with sentiment labels (negative, positive, neutral) and the URL.
    """
    try:
        # Validate URL
        reddit_pattern = r'https?://(?:www\.)?reddit\.com/r/.*?/comments/.*'
        x_pattern = r'https?://(?:www\.)?(?:x\.com|twitter\.com)/[^/]+/status/\d+'
        if not re.match(reddit_pattern, request.url) and not re.match(x_pattern, request.url):
            raise HTTPException(status_code=400, detail="Invalid Reddit URL format")
        
        result = generate_comprehensive_report(request.url)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Convert dict pain points to PainPoint models
        # Ensure pain_points is a list of dictionaries
        pain_points_raw = result["pain_points"]
        if not isinstance(pain_points_raw, list):
            raise HTTPException(status_code=500, detail="Invalid pain points format: expected list")
        
        pain_points = []
        for pp in pain_points_raw:
            # Handle case where pp might be a list (nested) or not a dict
            if isinstance(pp, list):
                # If it's a list, skip it or take the first item if it's a dict
                if pp and isinstance(pp[0], dict):
                    pp = pp[0]
                else:
                    continue
            elif not isinstance(pp, dict):
                # Skip non-dict items
                continue
            
            # Now pp should be a dict, safely extract values
            pain_points.append(
            PainPoint(
                description=pp.get("description", ""),
                sentiment=pp.get("sentiment", "neutral"),
                    quote=pp.get("quote", ""),
                    type=pp.get("type"),  # Include type field if present
                    commenters_count=pp.get("commenters_count", 0)  # Include commenters count
            )
            )
        
        return ComprehensiveReportResponse(
            pain_points=pain_points,
            pain_points_count=len(pain_points),
            url=result["url"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating comprehensive report: {str(e)}")

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
