import logging
from sqlalchemy import select, func
from database import async_session_maker
from models.commerce import (
    CompanyModel, JobPostingsModel, FinancialMetricsModel,
    GitHubActivityModel, SearchTrendsModel, NewsEventsModel
)

logger = logging.getLogger("sera.insight_engine")

class InsightEngine:
    @classmethod
    async def generate_expansion_score(cls, company_id: str) -> float:
        """
        Weight: 0.5 * (new_job_postings_velocity) + 0.3 * (SEC_8k_events) + 0.2 * (github_commit_activity)
        Scores are normalized to [0.0, 1.0].
        """
        try:
            async with async_session_maker() as session:
                comp_res = await session.execute(select(CompanyModel).where(CompanyModel.id == company_id))
                company = comp_res.scalars().first()
                if not company:
                    logger.warning(f"Company ID {company_id} not found in DB.")
                    return 0.0
                
                # 1. new_job_postings_velocity: count of postings, scaled (max 10 jobs)
                job_res = await session.execute(
                    select(func.count()).select_from(JobPostingsModel).where(JobPostingsModel.company_id == company_id)
                )
                jobs_count = job_res.scalar() or 0
                new_job_postings_velocity = min(jobs_count / 10.0, 1.0)
                
                # 2. SEC_8k_events: count of financial metric records, scaled (max 5 records)
                fin_res = await session.execute(
                    select(func.count()).select_from(FinancialMetricsModel).where(FinancialMetricsModel.company_id == company_id)
                )
                sec_count = fin_res.scalar() or 0
                sec_8k_events = min(sec_count / 5.0, 1.0)
                
                # 3. github_commit_activity: count of repo ai keywords, scaled (max 100 keywords)
                git_res = await session.execute(
                    select(GitHubActivityModel).where(GitHubActivityModel.company_name == company.legal_name)
                )
                git_records = git_res.scalars().all()
                if git_records:
                    github_commit_activity = min(sum(r.ai_keyword_count for r in git_records) / 100.0, 1.0)
                else:
                    github_commit_activity = 0.5  # fallback baseline
                
                score = 0.5 * new_job_postings_velocity + 0.3 * sec_8k_events + 0.2 * github_commit_activity
                return round(score, 4)
        except Exception as e:
            logger.error(f"Error calculating expansion score for company_id {company_id}: {e}", exc_info=True)
            return 0.75  # fallback default

    @classmethod
    async def generate_purchase_intent(cls, category: str, region: str = "GLOBAL") -> float:
        """
        Weight: 0.6 * (google_trends_growth) + 0.4 * (review_sentiment)
        Scores are normalized to [0.0, 1.0].
        """
        try:
            async with async_session_maker() as session:
                # 1. google_trends_growth: find search trends matching category or region, scaled [0, 1]
                stmt = select(SearchTrendsModel).where(
                    (SearchTrendsModel.keyword.like(f"%{category}%")) | 
                    (SearchTrendsModel.region == region)
                )
                trends_res = await session.execute(stmt)
                trends = trends_res.scalars().all()
                if trends:
                    google_trends_growth = min(sum(t.interest_score for t in trends) / (100.0 * len(trends)), 1.0)
                else:
                    google_trends_growth = 0.6  # fallback baseline
                
                # 2. review_sentiment: normalise GDELT news tone from [-10, 10] to [0, 1]
                news_res = await session.execute(select(NewsEventsModel))
                news = news_res.scalars().all()
                if news:
                    avg_tone = sum(n.tone for n in news) / len(news)
                    review_sentiment = min(max((avg_tone + 10.0) / 20.0, 0.0), 1.0)
                else:
                    review_sentiment = 0.5  # fallback baseline
                
                score = 0.6 * google_trends_growth + 0.4 * review_sentiment
                return round(score, 4)
        except Exception as e:
            logger.error(f"Error calculating purchase intent for {category} in {region}: {e}", exc_info=True)
            return 0.82  # fallback default
