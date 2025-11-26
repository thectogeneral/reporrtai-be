from pydantic import BaseModel, Field

class ExecutiveSummary(BaseModel):
    overview: str = Field(description="Brief overview of the product or company")
    initial_goals_or_kpis: str = Field(description="Initial goals or KPIs set at the beginning")
    performance_summary: str = Field(description="Overall performance summary including successes and challenges")


class CustomerInsightsAndPainPoints(BaseModel):
    initial_pain_points_identified: str = Field(description="Pain points identified at launch")
    pain_points_resolved_effectively: str = Field(description="How effectively pain points were addressed")
    new_or_emerging_pain_points: str = Field(description="New unmet needs or pain points that have emerged")
    user_feedback_and_sentiment_trends: str = Field(description="User sentiment trends and notable feedback")


class ProductPerformance(BaseModel):
    what_worked_well: str = Field(description="Successful features, UX, engagement")
    what_underperformed: str = Field(description="Underperforming features, design choices, or functionality")
    adoption_and_usage_trends: str = Field(description="Metrics or observed trends in adoption and engagement")


class BusinessAndMarketPerformance(BaseModel):
    revenue_vs_projections: str = Field(description="Revenue performance compared with projections")
    customer_acquisition_and_retention: str = Field(description="Acquisition and retention analysis")
    market_share_and_competitive_position: str = Field(description="Competitive landscape and market share performance")


class OperationalReview(BaseModel):
    execution_and_workflow_effectiveness: str = Field(description="Internal execution and workflows review")
    bottlenecks_or_inefficiencies: str = Field(description="Operational bottlenecks or inefficiencies identified")
    team_culture_and_alignment: str = Field(description="Team culture, alignment, and collaboration effectiveness")


class MarketingAndGrowthReview(BaseModel):
    channel_performance: str = Field(description="Performance of acquisition and marketing channels")
    conversion_funnel_analysis: str = Field(description="Conversion funnel performance and insights")
    brand_perception_and_community: str = Field(description="Brand sentiment and community engagement")


class TechnologyAndInfrastructureReview(BaseModel):
    stability_and_uptime: str = Field(description="System uptime, stability, and technical reliability")
    bugs_or_technical_debt: str = Field(description="Recurring bugs, performance issues, or technical debt")
    architecture_or_tech_stack_lessons: str = Field(description="Key lessons from architecture or tech stack decisions")


class FinancialOverview(BaseModel):
    revenue_breakdown: str = Field(description="Breakdown of revenue sources")
    cost_breakdown: str = Field(description="Breakdown of key costs")
    profitability_trends: str = Field(description="Profitability trends and patterns")
    key_financial_metrics: str = Field(description="Important financial indicators such as CAC, LTV, margin, etc.")


class LessonsLearned(BaseModel):
    key_wins: str = Field(description="Major wins during the period")
    mistakes_and_missteps: str = Field(description="Mistakes, missteps, and what went wrong")
    validated_assumptions: str = Field(description="Assumptions that were proven correct")
    invalidated_assumptions: str = Field(description="Assumptions that were disproven")


class StrategicAdjustmentsAndNextSteps(BaseModel):
    planned_improvements_or_pivots: str = Field(description="Improvements, pivots, or changes planned")
    areas_to_deprioritize_or_sunset: str = Field(description="Areas or features to reduce focus on or sunset")


class FutureRoadmap(BaseModel):
    short_term_3_to_6_months: str = Field(description="Short-term roadmap (3–6 months)")
    mid_term_6_to_12_months: str = Field(description="Mid-term roadmap (6–12 months)")
    long_term_1_to_3_years: str = Field(description="Long-term roadmap (1–3 years)")


class RisksAndMitigation(BaseModel):
    forward_risks: str = Field(description="Key risks moving forward")
    mitigation_strategies: str = Field(description="Recommended mitigation strategies")


class Conclusion(BaseModel):
    momentum_and_outlook: str = Field(description="Overall momentum and forward outlook")
    strategic_recommendations: str = Field(description="Strategic recommendations for the next phase")


class PerformanceReportOutput(BaseModel):
    executive_summary: ExecutiveSummary
    customer_insights_and_pain_points: CustomerInsightsAndPainPoints
    product_performance: ProductPerformance
    business_and_market_performance: BusinessAndMarketPerformance
    operational_review: OperationalReview
    marketing_and_growth_review: MarketingAndGrowthReview
    technology_and_infrastructure_review: TechnologyAndInfrastructureReview
    financial_overview: FinancialOverview
    lessons_learned: LessonsLearned
    strategic_adjustments_and_next_steps: StrategicAdjustmentsAndNextSteps
    future_roadmap: FutureRoadmap
    risks_and_mitigation: RisksAndMitigation
    conclusion: Conclusion
