from fastapi import APIRouter, Query
from core.entity_resolution import entity_registry

router = APIRouter(redirect_slashes=False)

@router.get("", include_in_schema=True)
@router.get("/", include_in_schema=False)
async def list_entities(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    all_entities = entity_registry.get_all()
    paginated = all_entities[offset:offset+limit]
    return {
        "total": len(all_entities),
        "limit": limit,
        "offset": offset,
        "entities": paginated
    }

@router.get("/{entity_id}")
async def get_entity(entity_id: str):
    entity = entity_registry.get_by_id(entity_id)
    
    if not entity:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity