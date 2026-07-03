from abc import ABC, abstractmethod

class EntityInterface(ABC):
    """Abstract interface for the Entity AI layer. Your team implements this."""
    @abstractmethod
    async def predict(self, entity_id: str, context: dict) -> dict:
        """Generate a causal prediction for the given entity."""
        ...
    @abstractmethod
    async def counterfactual(self, entity_id: str, intervention: dict) -> dict:
        """Simulate a counterfactual: what happens if we intervene?"""
        ...
    @abstractmethod
    async def get_causal_graph(self, entity_id: str) -> dict:
        """Return the causal graph for this entity's behavioral drivers."""
        ...