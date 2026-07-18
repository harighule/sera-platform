from fastapi import APIRouter, HTTPException
from services.graph_sync import GraphSyncService

router = APIRouter(prefix="/api/graph", tags=["graph"])

@router.post("/sync")
async def sync_graph_database():
    """
    Manually triggers property graph synchronization from PostgreSQL to Neo4j.
    """
    try:
        metrics = await GraphSyncService.sync_all_entities()
        return {
            "status": "success",
            "message": "Graph database successfully synchronized.",
            "metrics": metrics
        }
    except ConnectionError as conn_err:
        raise HTTPException(status_code=503, detail=str(conn_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to synchronize graph database: {e}")
