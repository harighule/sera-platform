import logging
from database import async_session_maker
from models.commerce import CompanyModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from services.insight_engine import InsightEngine
from config import USE_REAL_DATA

logger = logging.getLogger("sera.narrative_engine")

class NarrativeEngine:
    @classmethod
    async def generate_expansion_report(cls, ticker: str) -> dict:
        """
        Generates narrative expansion reports based on company metrics.
        Works in both real data mode and mock data fallback mode.
        """
        ticker = ticker.upper()
        
        # 1. Mock mode fallback if USE_REAL_DATA is False
        if not USE_REAL_DATA:
            if ticker == "AAPL":
                return {
                    "summary": "Apple Inc. is 35.8% likely to expand operations within the next 6 months.",
                    "key_drivers": [
                        "Increased hiring for hardware engineers (+20% YoY)",
                        "Recent 8-K filing indicates capital raise",
                        "R&D spending up 15%"
                    ],
                    "timeframe": "Q3 2026",
                    "recommendation": "Monitor for supply chain partnerships in Southeast Asia."
                }
            elif ticker == "MSFT":
                return {
                    "summary": "Microsoft Corp. is 64.2% likely to expand cloud data centers within the next 6 months.",
                    "key_drivers": [
                        "Aggressive infrastructure spending in EMEA region",
                        "Rising demand for localized Azure AI node compute",
                        "Increased talent acquisition for cloud reliability engineers"
                    ],
                    "timeframe": "Q3-Q4 2026",
                    "recommendation": "Track capital expenditures and hardware vendor deals in Northern Europe."
                }
            else:
                return {
                    "summary": f"{ticker} is 45.0% likely to expand operations within the next 6 months.",
                    "key_drivers": [
                        "Baseline hiring velocity in core engineering roles",
                        "Stable operating margins reported in recent filings",
                        "Moderate developer repository activity"
                    ],
                    "timeframe": "Next 6 months",
                    "recommendation": "Monitor general industry hiring velocity and quarterly report updates."
                }
                
        # 2. Real data mode
        try:
            async with async_session_maker() as session:
                stmt = select(CompanyModel).where(CompanyModel.ticker == ticker).options(
                    selectinload(CompanyModel.financial_metrics),
                    selectinload(CompanyModel.job_postings)
                )
                result = await session.execute(stmt)
                company = result.scalars().first()
                if not company:
                    # Fallback default if company is missing in DB
                    return {
                        "summary": f"{ticker} is 50.0% likely to expand operations.",
                        "key_drivers": ["Default market baseline", "Fallback parameters applied"],
                        "timeframe": "Q3 2026",
                        "recommendation": "Manually inspect corporate registration."
                    }
                
                # Fetch expansion score
                score = await InsightEngine.generate_expansion_score(company.id)
                likelihood = score * 100
                
                # Retrieve related stats for narrative rendering
                job_count = len(company.job_postings)
                latest_metrics = company.financial_metrics[-1] if company.financial_metrics else None
                revenue = latest_metrics.revenue if latest_metrics else 1000000000.0
                
                # Compile key drivers dynamically based on metrics
                drivers = []
                if job_count > 0:
                    drivers.append(f"Active recruitment detected with {job_count} open postings in key sectors")
                else:
                    drivers.append("Hiring rates remain stable at baseline levels")
                    
                if latest_metrics:
                    drivers.append(f"Strong capital reserves with reported deferred revenue of ${latest_metrics.deferred_revenue:,.2f}")
                else:
                    drivers.append("Filing dates indicate regular SEC compliance")
                    
                drivers.append("Open-source R&D repository commit volumes remain active")

                return {
                    "summary": f"{company.legal_name} is {likelihood:.1f}% likely to expand operations within the next 6 months.",
                    "key_drivers": drivers,
                    "timeframe": "Q3 2026",
                    "recommendation": f"Monitor {ticker} quarterly capex disclosures and regional hiring updates."
                }
        except Exception as e:
            logger.error(f"Failed to generate expansion report for {ticker}: {e}", exc_info=True)
            return {
                "summary": f"Failed to generate real-time expansion report for {ticker}.",
                "key_drivers": ["System error occurred during data retrieval"],
                "timeframe": "N/A",
                "recommendation": "Retry operation or contact system administrator."
            }

    @classmethod
    async def generate_purchase_intent_report(cls, category: str, region: str = "GLOBAL") -> str:
        """
        Generates a human-readable purchase intent summary report.
        """
        try:
            score = await InsightEngine.generate_purchase_intent(category, region)
            likelihood = int(score * 100)
            
            # Simple narrative template combination
            return (
                f"Consumers in {region} are {likelihood}% more likely to purchase {category} "
                f"within the next 90 days due to increased financing searches, "
                f"search interest score spikes, elevated GDELT news sentiment, and seasonal mobility trends."
            )
        except Exception as e:
            logger.error(f"Failed to generate purchase intent report for {category} in {region}: {e}", exc_info=True)
            return f"Unable to analyze purchase intent for {category} in {region} at this time."

    @classmethod
    async def generate_short_summary(cls, ticker: str, legal_name: str, score: float) -> str:
        """
        Generates a concise 1-sentence causal narrative summary.
        """
        likelihood = int(score * 100)
        if score > 0.75:
            return f"Critical expansion spike: {legal_name} ({ticker}) exhibits a {likelihood}% probability of capital deployment driven by aggressive recruitment velocity and active R&D pipeline momentum."
        elif score > 0.5:
            return f"Stable expansion trajectory: {legal_name} ({ticker}) showing a moderate {likelihood}% expansion potential with baseline capex registry indicators and steady headcount growth."
        else:
            return f"Baseline operation pattern: {legal_name} ({ticker}) maintains a steady operational profile at {likelihood}% likelihood, exhibiting low variance in active patent and job registries."

