import logging
import math
import requests
import asyncio
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sqlalchemy import select, delete, desc, func
import yfinance as yf
from database import async_session_maker
from models.commerce import (
    CompanyModel, JobPostingsModel, NewsEventsModel, TickerPriorityCacheModel
)
from config import USE_REAL_DATA

logger = logging.getLogger("sera.ticker_discovery")

DEFAULT_UNIVERSE = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"ticker": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
    {"ticker": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare"},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Financial Services"},
    {"ticker": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Defensive"},
    {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Financial Services"},
    {"ticker": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical"},
    {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
    {"ticker": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
    {"ticker": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Defensive"},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Defensive"},
    {"ticker": "BAC", "name": "Bank of America Corp.", "sector": "Financial Services"},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
]

class DynamicTickerDiscovery:
    @classmethod
    def fetch_all_universe(cls) -> list[dict]:
        """
        Downloads/scrapes S&P 500 constituents dynamically.
        Returns a list of dicts: [{"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"}, ...]
        """
        # Try Wikipedia pandas scraper (extremely stable, unblocked)
        try:
            import pandas as pd
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            if tables and len(tables) > 0:
                df = tables[0]
                results = []
                for _, row in df.iterrows():
                    ticker = str(row['Symbol']).replace('.', '-')
                    name = str(row['Security'])
                    sector = str(row['GICS Sector'])
                    results.append({"ticker": ticker, "name": name, "sector": sector})
                if len(results) > 100:
                    logger.info(f"Successfully fetched {len(results)} tickers via Wikipedia pandas.")
                    return results
        except Exception as e:
            logger.warning(f"Wikipedia pandas scraper failed: {e}. Trying BeautifulSoup...")

        # Try Wikipedia BeautifulSoup fallback
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'lxml')
                table = soup.find('table', {'id': 'constituents'})
                if table:
                    results = []
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            ticker = cols[0].text.strip().replace('.', '-')
                            name = cols[1].text.strip()
                            sector = cols[3].text.strip()
                            results.append({"ticker": ticker, "name": name, "sector": sector})
                    if len(results) > 100:
                        logger.info(f"Successfully fetched {len(results)} tickers via Wikipedia bs4.")
                        return results
        except Exception as e:
            logger.warning(f"Wikipedia bs4 scraping failed: {e}. Trying StockAnalysis...")

        # Try StockAnalysis BeautifulSoup scraper
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get('https://stockanalysis.com/list/sp-500-stocks/', headers=headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'lxml')
                table = soup.find('table')
                if table:
                    results = []
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            ticker = cols[1].text.strip().replace('.', '-')
                            name = cols[2].text.strip()
                            sector = cols[3].text.strip() if len(cols) > 3 else "Technology"
                            results.append({"ticker": ticker, "name": name, "sector": sector})
                    if len(results) > 100:
                        logger.info(f"Successfully fetched {len(results)} tickers via StockAnalysis.")
                        return results
        except Exception as e:
            logger.error(f"StockAnalysis scraping failed: {e}")

        logger.warning("All scraping universes failed. Falling back to default built-in top tickers.")
        return DEFAULT_UNIVERSE

    @classmethod
    async def calculate_relevance_score(cls, ticker: str, db_session) -> float:
        """
        Weights GDELT news volume, job momentum, and yfinance market capitalization.
        Weight: 0.4 * news_count + 0.3 * job_count + 0.3 * market_cap_normalized
        """
        try:
            # 1. Local News Events mentions count
            news_stmt = select(func.count()).select_from(NewsEventsModel).where(
                NewsEventsModel.title.like(f"%{ticker}%")
            )
            news_count = (await db_session.execute(news_stmt)).scalar() or 0

            # 2. Local Job Postings count
            job_stmt = select(func.count()).select_from(JobPostingsModel).where(
                JobPostingsModel.company_id.like(f"%{ticker}%")
            )
            job_count = (await db_session.execute(job_stmt)).scalar() or 0

            # 3. Market Cap via yfinance with fallback base
            # To speed up and prevent yfinance network delays during bulk calculation,
            # we default base marketCap values unless it's a top company where we fetch it dynamically.
            market_cap = 10_000_000_000 # 10B default
            if ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
                try:
                    # Run yfinance request in executor to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    ticker_obj = yf.Ticker(ticker)
                    info = await loop.run_in_executor(None, lambda: ticker_obj.info)
                    market_cap = info.get("marketCap", market_cap)
                    
                    # Also append yfinance news mentions
                    yf_news = info.get("news", [])
                    news_count += len(yf_news)
                except Exception as yf_err:
                    logger.debug(f"Failed to fetch yfinance info for {ticker}: {yf_err}")

            # Normalize values
            news_norm = min(news_count / 10.0, 1.0)
            job_norm = min(job_count / 5.0, 1.0)
            market_cap_norm = min(market_cap / 500_000_000_000, 1.0) # Normalized to 500B cap

            score = 0.4 * news_norm + 0.3 * job_norm + 0.3 * market_cap_norm
            return round(score, 4)
        except Exception as e:
            logger.error(f"Error calculating relevance score for {ticker}: {e}")
            return 0.25

    @classmethod
    async def get_top_n_tickers(cls, n: int = 50) -> list[dict]:
        """
        Retrieves top n companies sorted descending by relevance score.
        Checks for cached records less than 4 hours old.
        """
        now = datetime.utcnow()
        cache_threshold = now - timedelta(hours=4)

        try:
            async with async_session_maker() as session:
                # Check priority cache status
                cache_stmt = select(TickerPriorityCacheModel).where(
                    TickerPriorityCacheModel.last_updated >= cache_threshold
                ).order_by(desc(TickerPriorityCacheModel.relevance_score)).limit(n)
                
                cache_res = await session.execute(cache_stmt)
                cached_records = cache_res.scalars().all()
                if len(cached_records) >= n:
                    logger.info(f"Returning {len(cached_records)} tickers from database priority cache.")
                    return [
                        {
                            "ticker": r.ticker,
                            "name": r.company_name,
                            "sector": r.sector,
                            "relevance_score": r.relevance_score
                        }
                        for r in cached_records
                    ]

                # Fallback to fetching fresh universe
                logger.info("Priority cache is stale or empty. Discovering fresh universe...")
                universe = cls.fetch_all_universe()
                scored_universe = []
                
                # To prevent rate-limiting and timeouts, score first 100 tickers and seed default for others
                for idx, item in enumerate(universe):
                    ticker = item["ticker"]
                    name = item["name"]
                    sector = item["sector"]
                    
                    if idx < 100 or ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
                        score = await cls.calculate_relevance_score(ticker, session)
                    else:
                        # Baseline weight mapping
                        score = round(0.1 + (0.1 if sector == "Technology" else 0.05), 3)
                        
                    scored_universe.append({
                        "ticker": ticker,
                        "name": name,
                        "sector": sector,
                        "relevance_score": score
                    })

                # Sort by score descending
                scored_universe.sort(key=lambda x: x["relevance_score"], reverse=True)
                top_tickers = scored_universe[:n]
                
                # Save top 100 to priority cache table
                await session.execute(delete(TickerPriorityCacheModel))
                for item in scored_universe[:100]:
                    session.add(TickerPriorityCacheModel(
                        ticker=item["ticker"],
                        company_name=item["name"],
                        sector=item["sector"],
                        relevance_score=item["relevance_score"],
                        last_updated=now
                    ))
                await session.commit()
                
                logger.info(f"Discovered {len(universe)} tickers. Top {n} prioritized: {[t['ticker'] for t in top_tickers]}")
                return top_tickers
        except Exception as e:
            logger.error(f"Error in dynamic ticker discovery flow: {e}", exc_info=True)
            # Safe database fallback: return top default list
            return DEFAULT_UNIVERSE[:n]
