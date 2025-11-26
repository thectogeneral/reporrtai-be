
# Prompt for idea generation (multiple ideas with structured data)
idea_template_single2 = """You are an innovative product strategist and system designer. 
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
idea_multiple_single2 = """You are an innovative product strategist and system designer. 
        Given these pain points:
        {pain_points}

        Propose **3-5** app ideas app idea or digital solution (name it) that directly addresses these issues. 
        Then provide a **comprehensive and actionable product roadmap** covering the following:

        1. **Product Overview**
        - App Name
        - One-sentence description of what the solution/product/business/app is
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

performance_review_template2 = """You are a strategic product analyst and business reviewer. 
        Given this context:
        {thread_text}

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

        Your response should read like a **professional product performance report** — data-driven, reflective, and actionable."""

