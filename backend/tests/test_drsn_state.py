import sys
import os
import pytest
from entity_interface.live_entity import LiveEntity
from core.entity_resolution import entity_registry

@pytest.mark.anyio
async def test_drsn_state_and_reset():
    entity_registry.entities["drsn-t"] = {
        "id": "drsn-t", "name": "drsn", "domain": "financial", "status": "pre-transition",
        "entropy": 2.0, "event_count": 100, "alert_count": 10,
    }
    le = LiveEntity()
    
    # 1. Assert call count starts at 0 and increments on predict()
    assert le.stats.get("drsn_call_count", 0) == 0
    p1 = await le.predict("drsn-t", {})
    assert le.stats.get("drsn_call_count") == 1
    p2 = await le.predict("drsn-t", {})
    assert le.stats.get("drsn_call_count") == 2
    
    # 2. Call reset on a DRSN instance with accumulated state
    drsn = le.drsn
    # Ensure it ran and has non-rest state
    assert any(n.state.V != n.V_rest for n in drsn.nodes) or any(n.state.theta != -55.0 for n in drsn.nodes)
    
    drsn.reset()
    # Assert reset returns to default
    for n in drsn.nodes:
        assert n.state.V == n.V_rest
        assert n.state.theta == -55.0
        assert n.state.spike_count == 0
        assert n.state.last_spike_t == -1000.0
        assert n.t == 0.0

    # 3. Assert active_nodes and total_spikes are present
    assert "drsn_active_nodes" in p1
    assert "drsn_total_spikes" in p1
