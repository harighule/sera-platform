import os
import sys
import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Set path so we can import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import async_session_maker
from models.commerce import CompanyModel
from sqlalchemy import select

async def main():
    load_dotenv()
    
    # 1. Fetch companies from PostgreSQL/SQLite
    print("Fetching companies from database...")
    async with async_session_maker() as session:
        res = await session.execute(select(CompanyModel))
        companies = res.scalars().all()
    print(f"Fetched {len(companies)} companies.")

    # 2. Connect to Neo4j
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "SeraGraphPass2026!")
    
    # Local dev fallback: if host is 'neo4j', but we are running outside Docker, try 'localhost'
    if "neo4j" in neo4j_uri and not os.path.exists("/.dockerenv"):
        alternative_uri = neo4j_uri.replace("neo4j", "localhost")
        print(f"Running outside Docker container. Will try local fallback URI: {alternative_uri}")
    else:
        alternative_uri = None
        
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    try:
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            driver.verify_connectivity()
        except Exception as conn_err:
            if alternative_uri:
                print(f"Failed to connect to primary URI ({conn_err}). Trying fallback URI {alternative_uri}...")
                driver = GraphDatabase.driver(alternative_uri, auth=(neo4j_user, neo4j_password))
                driver.verify_connectivity()
            else:
                raise conn_err
                
        with driver.session() as neo_session:
            print("Successfully connected to Neo4j. Syncing nodes...")
            
            # Batch sync to prevent slow individual roundtrips
            tx = neo_session.begin_transaction()
            try:
                for idx, c in enumerate(companies):
                    tx.run(
                        """
                        MERGE (comp:Company {ticker: $ticker})
                        SET comp.name = $name,
                            comp.legal_name = $name,
                            comp.sector = $sector,
                            comp.id = $id
                        """,
                        ticker=c.ticker,
                        name=c.legal_name,
                        sector=c.sector or "Technology",
                        id=c.id
                    )
                    if (idx + 1) % 1000 == 0:
                        tx.commit()
                        print(f"Committed {idx + 1} companies to Neo4j...")
                        tx = neo_session.begin_transaction()
                tx.commit()
                print("Completed Neo4j graph sync successfully!")
            except Exception as tx_err:
                tx.rollback()
                print(f"Transaction failed: {tx_err}")
                raise tx_err
        driver.close()
    except Exception as e:
        print(f"ERROR syncing with Neo4j: {e}")

if __name__ == "__main__":
    asyncio.run(main())
