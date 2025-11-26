from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# USER JOURNEY + PAIN POINT MAPPING
# ─────────────────────────────────────────

class UserJourneyStep(BaseModel):
    step: str = Field(description="The name of the step in the user journey")
    touchpoint: str = Field(description="Where the user interacts at this step")
    emotional_state: str = Field(description="How the user feels at this step")


class Feature(BaseModel):
    feature_name: str = Field(description="Name of the feature")
    description: str = Field(description="Short explanation of what the feature does")
    is_mvp_feature: bool = Field(description="Whether this feature is included in the MVP")


class PainPointMapping(BaseModel):
    feature_name: str = Field(description="The feature addressing pain points")
    solves_pain_points: List[str] = Field(description="Pain points addressed by this feature")
    value_created: str = Field(description="Description of the value produced by solving the pain points")


# ─────────────────────────────────────────
# COMPETITIVE LANDSCAPE
# ─────────────────────────────────────────

class Competitor(BaseModel):
    name: str = Field(description="Competitor or similar product")
    strengths: str = Field(description="What they do well")
    weaknesses: str = Field(description="Where they fall short")
    comparison_summary: str = Field(description="Summary of how the proposed app compares")


class CompetitiveLandscape(BaseModel):
    competitors: List[Competitor]
    unique_differentiators: str = Field(description="Key elements that set this app apart")
    market_gaps: str = Field(description="Gaps in the market the product will fill")


# ─────────────────────────────────────────
# ROADMAP STAGES
# ─────────────────────────────────────────

class RoadmapStage(BaseModel):
    features: List[str] = Field(description="Features included in this stage")
    goals: str = Field(description="Primary goals for this stage")
    metrics: List[str] = Field(description="Key metrics used to measure success in this stage")


class ProductRoadmap(BaseModel):
    mvp: RoadmapStage
    v1: RoadmapStage
    v2: RoadmapStage
    full_product: RoadmapStage


# ─────────────────────────────────────────
# INDUSTRY + USERS
# ─────────────────────────────────────────

class Industry(BaseModel):
    sector: str = Field(description="Industry category the app belongs to")
    problem_solved: str = Field(description="Description of the core problem the app solves")
    target_audience: str = Field(description="The audience the product is for")
    business_stage: str = Field(description="Idea stage, prototype, early-stage startup, etc.")


class TargetUsers(BaseModel):
    primary_users: str = Field(description="Main intended users")
    secondary_users: str = Field(description="Secondary or optional users")
    motivations: str = Field(description="What drives these users to adopt the product")
    demographics_and_behavior: str = Field(description="Key traits, habits, or contexts of use")


# ─────────────────────────────────────────
# TECH + COMPLIANCE
# ─────────────────────────────────────────

class TechStack(BaseModel):
    frontend: str
    backend: str
    database: str
    mobile: Optional[str] = None
    other: Optional[str] = None


class CICDAndMonitoring(BaseModel):
    pipeline: str = Field(description="CI/CD tools or processes in use")
    monitoring_tools: str = Field(description="Tools used for error tracking and monitoring")


class Compliance(BaseModel):
    kyc_aml: bool = Field(description="Whether KYC/AML is required")
    gdpr: bool = Field(description="Whether GDPR compliance is required")
    data_privacy: str = Field(description="Notes on data privacy requirements")


class TechnicalImplementation(BaseModel):
    tech_stack: TechStack
    architecture: str = Field(description="Monolith, microservices, serverless, etc.")
    required_integrations: List[str] = Field(description="APIs or services required")
    scalability_considerations: str = Field(description="How the system will scale")
    deployment_environment: str = Field(description="AWS, GCP, Vercel, etc.")
    cicd_and_monitoring: CICDAndMonitoring
    compliance: Compliance


# ─────────────────────────────────────────
# BUSINESS MODEL
# ─────────────────────────────────────────

class MonetizationForecast(BaseModel):
    user_volume: str = Field(description="Expected number of users")
    pricing: str = Field(description="Pricing for the model")
    projected_revenue: str = Field(description="MRR/ARR estimate or equivalent")


class BusinessModel(BaseModel):
    model_type: str = Field(description="SaaS, subscription, freemium, etc.")
    pricing_strategy: str = Field(description="How the product will be priced")
    customer_acquisition: str = Field(description="How new users will be acquired")
    customer_retention: str = Field(description="How existing users will be retained")
    monetization_forecast: MonetizationForecast


class MonetizationStrategy(BaseModel):
    strategy_name: str
    description: str
    ideal_stage: str = Field(description="Stage where this strategy is most appropriate: MVP, V1, scaling")


# ─────────────────────────────────────────
# GO-TO-MARKET + METRICS
# ─────────────────────────────────────────

class GoToMarket(BaseModel):
    launch_sequence: List[str] = Field(description="Beta → launch → expansion steps")
    growth_channels: List[str] = Field(description="Marketing or growth channels")
    early_adopter_tactics: str = Field(description="Ways to attract first users")
    community_building: str = Field(description="How to build a community around the app")
    content_marketing_timeline: List[str] = Field(description="Content themes by week or month")


class SuccessMetrics(BaseModel):
    mvp: List[str]
    v1: List[str]
    v2: List[str]
    overall_kpis: List[str]


# ─────────────────────────────────────────
# TEAM + LONG-TERM
# ─────────────────────────────────────────

class TeamRole(BaseModel):
    role: str = Field(description="Name of the role")
    responsibilities: str = Field(description="What this role is responsible for")


class LongTermOpportunities(BaseModel):
    feature_expansion: List[str]
    integrations: List[str]
    new_markets: List[str]
    strategic_partnerships: List[str]
    five_year_vision: str = Field(description="High-level view of where the app is going long-term")


# ─────────────────────────────────────────
# MVP SCOPE
# ─────────────────────────────────────────

class MVPScope(BaseModel):
    core_value: str = Field(description="Smallest version that provides meaningful value")
    included_workflows: List[str] = Field(description="User journeys included in MVP")
    unique_selling_point: str = Field(description="USP of the MVP")
    growth_potential: str = Field(description="Potential for expansion")
    market_size_estimate: str = Field(description="Market TAM/SAM/SOM or a rough estimate")
    excluded_from_mvp: List[str] = Field(description="What is intentionally not included")

class ExtractedPainPoint(BaseModel):
    title: str
    number_of_users: int
    category: str
    quotes: List[str]
# ─────────────────────────────────────────
# MAIN APP MODEL
# ─────────────────────────────────────────


class ChallengesAndRisks(BaseModel):
    risks: List[str] = Field(description="List of major risks")
    mitigation: List[str] = Field(description="How to mitigate each risk")
    competitor_lessons: str = Field(description="Insights from competitor failures or weaknesses")

class AppIdea(BaseModel):
    app_name: str
    one_sentence_description: str
    problem_summary: str
    value_proposition: str
    vision: str
    industry: Industry
    target_users: TargetUsers
    competitive_landscape: CompetitiveLandscape
    core_features: List[Feature]
    pain_point_mapping: List[PainPointMapping]
    mvp_scope: MVPScope
    product_roadmap: ProductRoadmap
    user_experience: List[UserJourneyStep]
    technical_implementation: TechnicalImplementation
    business_model: BusinessModel
    monetization_strategies: List[MonetizationStrategy]
    go_to_market_strategy: GoToMarket
    success_metrics: SuccessMetrics
    challenges_and_risks: ChallengesAndRisks
    team_and_skills: List[TeamRole]
    long_term_opportunities: LongTermOpportunities

class AppIdeaOutput(BaseModel):
    app_ideas: List[AppIdea]

    class Config:
        extra = "ignore"
