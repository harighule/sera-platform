from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from services.citation_service import CitationService

router = APIRouter(prefix="/api/citation", tags=["citation"])

class TrackQueryRequest(BaseModel):
    query_text: str
    target_entity_id: str = ""
    target_entity_name: str

@router.post("/track")
async def track_query(req: TrackQueryRequest):
    try:
        res = await CitationService.track_query(
            query_text=req.query_text,
            target_entity_id=req.target_entity_id or "",
            target_entity_name=req.target_entity_name
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tracked")
async def get_tracked_queries():
    try:
        queries = await CitationService.get_tracked_queries()
        return queries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run/{query_id}")
async def run_citation_check(query_id: str):
    """Force-run a citation check for the given query ID."""
    try:
        import uuid
        qid = uuid.UUID(query_id)
        await CitationService.update_citations(qid)
        # Return latest version
        queries = await CitationService.get_tracked_queries()
        match = next((q for q in queries if q["id"] == query_id), None)
        if match:
            return match
        return {"status": "updated", "query_id": query_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rate")
async def get_entity_citation_rate(entity_name: str = Query(..., description="Entity name to compute citation rate for")):
    """Compute citation rate for a named entity across all tracked queries."""
    try:
        queries = await CitationService.get_tracked_queries()
        entity_queries = [q for q in queries if q.get("target_entity_name", "").lower() == entity_name.lower()]
        if not entity_queries:
            return {
                "entity_name": entity_name,
                "tracked_queries": 0,
                "total_citations": 0,
                "avg_share_of_voice": 0.0,
                "max_share_of_voice": 0.0,
                "queries": []
            }
        total_cit = sum(q.get("citation_count", 0) for q in entity_queries)
        avg_sov = round(sum(q.get("share_of_voice", 0.0) for q in entity_queries) / len(entity_queries), 2)
        max_sov = round(max(q.get("share_of_voice", 0.0) for q in entity_queries), 2)
        return {
            "entity_name": entity_name,
            "tracked_queries": len(entity_queries),
            "total_citations": total_cit,
            "avg_share_of_voice": avg_sov,
            "max_share_of_voice": max_sov,
            "queries": entity_queries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
