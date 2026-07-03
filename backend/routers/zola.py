from fastapi import APIRouter, HTTPException
from core.entity_resolution import entity_registry
from entity_interface.mock_entity import MockEntity
from entity_interface.live_entity import LiveEntity
from entity_interface.axiom_compression import analyse_kronos_model, compress_kronos_model
from entity_interface.kronos.kronos_training import GodelLoop, KRONOSTrainer
from config import ENTITY_MODE
import asyncio
import random
import time
import torch
import json
from datetime import datetime
from database import async_session_maker
from models.db_models import PredictionModel

try:
    from entity_interface.kronos.orchestrator import KRONOSOrchestrator
    kronos_available = True
except ImportError:
    kronos_available = False

router = APIRouter(prefix="/api/zola", tags=["zola"])

# Dynamically instantiate the entity layer based on config
if ENTITY_MODE == "live":
    entity_ai = LiveEntity()
else:
    entity_ai = MockEntity()

# Godel Loop state — persists across requests within a single server process
_godel_loop = None
_godel_results = []
_best_evolved_config = {}

# Optimize-step counter for auto-triggering Godel generations
_optimize_call_counter = 0

async def save_prediction_to_db(prediction: dict):
    try:
        async with async_session_maker() as session:
            entity_id = prediction["entity_id"]
            prediction_type = prediction.get("type", "behavioral")
            confidence = prediction.get("confidence", 0.0)
            predicted_outcome = prediction.get("prediction", "")
            causal_factors = json.dumps(prediction.get("causal_factors", []))
            timestamp = datetime.utcnow()

            row = PredictionModel(
                entity_id=entity_id,
                transition_type=prediction.get("transition_type", prediction_type),
                causal_mechanism=prediction.get("causal_mechanism", predicted_outcome),
                optimal_intervention=prediction.get("optimal_intervention", ""),
                success_probability=prediction.get("success_probability", confidence),
                recommended_timing=prediction.get("recommended_timing", ""),
                consequence_chain=prediction.get("consequence_chain", json.loads(causal_factors))
            )
            session.add(row)
            await session.commit()
    except Exception as e:
        print(f"[ZOLA] DB prediction save failed: {e}")

@router.get("/predictions")
async def get_predictions():
    entities = [e for e in entity_registry.get_all() if e["status"] == "pre-transition"][:8]
    predictions = []

    for entity in entities:
        # Pass entropy explicitly in context
        prediction = await entity_ai.predict(entity["id"], {"entropy": entity["entropy"]})
        prediction["entity_name"] = entity["name"]
        prediction["domain"] = entity["domain"]
        predictions.append(prediction)
        asyncio.create_task(save_prediction_to_db(prediction))
    
    return predictions

@router.get("/status")
async def get_entity_status():
    """Retrieve current operational statistics and parameter counts for the Entity AI Layer."""
    if isinstance(entity_ai, LiveEntity):
        actual_stored_params = sum(p.numel() for p in entity_ai.model.parameters())
        return {
            "entity_mode": "live",
            "stats": entity_ai.stats,
            "actual_stored_params": actual_stored_params,
            "wave_basis_size_kb": entity_ai.stats["wave_basis_size_kb"],
            "architecture_summary": {
                "representation": "Continuous sinusoidal basis (CIFN)",
                "layer1": "CIFNLinear(8\u219216, basis=128)",
                "layer2": "CIFNLinear(16\u219215, basis=128)",
                "storage_model": "Wave parameters only \u2014 weight matrix computed on forward pass"
            }
        }
    else:
        # Default stats for mock mode to show placeholder numbers
        return {
            "entity_mode": "mock",
            "stats": {
                "virtual_parameters": 13_000_000_000,
                "wave_basis_size_kb": 0.0,
                "backprop_steps": 0,
                "latest_loss": 0.0,
                "latest_grad_norm": 0.0,
                "facts_crawled": 0,
                "self_evolution_cycles": 0,
                "pending_patches": [],
                "approved_patches": []
            },
            "actual_stored_params": 0,
            "wave_basis_size_kb": 0.0,
            "architecture_summary": {
                "representation": "Mock mode",
                "layer1": "Mock mode",
                "layer2": "Mock mode",
                "storage_model": "Mock mode"
            }
        }

@router.post("/learn")
async def trigger_cyberspace_learning():
    """Trigger background cyberspace crawlers to ingest new facts and scale weights."""
    if not isinstance(entity_ai, LiveEntity):
        raise HTTPException(status_code=400, detail="Cyberspace learning is only available in live ENTITY_MODE.")
    result = await entity_ai.trigger_cyberspace_learning()
    return result

@router.post("/evolve/propose")
async def propose_evolution_patch():
    """Generate a self-evolution code rewrite patch for validation."""
    if not isinstance(entity_ai, LiveEntity):
        raise HTTPException(status_code=400, detail="Self-evolution is only available in live ENTITY_MODE.")
    patch = entity_ai.propose_self_evolution_patch()
    return patch

@router.post("/evolve/validate/{patch_id}")
async def validate_evolution_patch(patch_id: int):
    """Sandbox compile the proposed code changes to verify zero regression."""
    if not isinstance(entity_ai, LiveEntity):
        raise HTTPException(status_code=400, detail="Sandbox validation is only available in live ENTITY_MODE.")
    success = entity_ai.validate_patch_sandbox(patch_id)
    return {"status": "success" if success else "failed", "verified": success}

@router.post("/evolve/approve/{patch_id}")
async def approve_evolution_patch(patch_id: int):
    """Approve and dynamically apply the verified patch."""
    if not isinstance(entity_ai, LiveEntity):
        raise HTTPException(status_code=400, detail="Patch approval is only available in live ENTITY_MODE.")
    success = entity_ai.approve_patch(patch_id)
    if not success:
        raise HTTPException(status_code=404, detail="Patch not found or failed application.")
    return {"status": "success", "applied": True}

@router.post("/ingest/csv")
async def ingest_csv():
    """Ingest transactions from the sample CSV file to trigger entropy spikes and alerts."""
    import csv
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_transactions.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Sample transaction CSV file not found.")
        
    try:
        from core.entropy_engine import entropy_engine
        processed_count = 0
        spikes_triggered = 0
        entities_updated = set()
        
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity_id = row["entity_id"]
                name = row["name"]
                event_type = row["event_type"]
                protocol = row["protocol"]
                
                # Dynamic registration of the user if not exists
                if entity_id not in entity_registry.entities:
                    entity_registry.entities[entity_id] = {
                        "id": entity_id,
                        "name": name,
                        "domain": "financial",
                        "status": "stable",
                        "entropy": 0.5,
                        "event_count": 0,
                        "alert_count": 0
                    }
                
                # Ingest into entropy engine
                metrics = entropy_engine.ingest(entity_id, event_type, protocol)
                
                # Update in registry
                entity_registry.update_entropy(entity_id, metrics["entropy"], metrics["alert_triggered"], metrics.get("z_score", 0.0))
                
                entities_updated.add(entity_id)
                processed_count += 1
                if metrics["alert_triggered"]:
                    spikes_triggered += 1
                    
        return {
            "status": "success",
            "events_processed": processed_count,
            "entities_updated_count": len(entities_updated),
            "alerts_triggered": spikes_triggered,
            "entities": list(entities_updated)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest CSV: {str(e)}")

@router.post("/kronos/optimize")
async def kronos_optimize():
    """Run a single internal backprop training step on the live Entity AI model and return metrics."""
    global _optimize_call_counter
    _optimize_call_counter += 1

    if not isinstance(entity_ai, LiveEntity):
        raise HTTPException(status_code=400, detail="Requires ENTITY_MODE=live")

    start = time.perf_counter()
    features = torch.randn(1, 8)
    target_prob = random.uniform(0.60, 0.90)
    entity_ai._run_internal_training_step(features, target_prob)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if _optimize_call_counter % 50 == 0 and _godel_loop is not None:
        asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(
                None, _godel_loop.step_generation
            )
        )
        print(f"[GODEL] Auto-triggered at optimize step {_optimize_call_counter}")

    return {
        "status": "success",
        "backprop_steps": entity_ai.stats["backprop_steps"],
        "latest_loss": entity_ai.stats["latest_loss"],
        "latest_grad_norm": entity_ai.stats["latest_grad_norm"],
        "latency_ms": latency_ms,
        "wave_basis_size_kb": entity_ai.stats["wave_basis_size_kb"],
        "virtual_parameters": entity_ai.stats["virtual_parameters"],
        "drsn_active_nodes": entity_ai.stats.get("drsn_total_spikes", 0),
        "architecture_layers": entity_ai.stats.get("architecture_layers", []),
        "architecture": {
            "layer1": "CIFNLinear(8→16, basis=128)",
            "layer2": "CIFNLinear(16→15, basis=128)",
            "actual_trainable_params": sum(p.numel() for p in entity_ai.model.parameters()),
            "weight_field_representation": "Continuous sinusoidal basis (compact)"
        }
    }

@router.get("/kronos/status")
async def get_kronos_status():
    return {
        "available": kronos_available,
        "current_phase": 1,
        "phase_label": "13B → 130B (Kronecker Width Expansion)",
        "phases": [
            {"phase": 1, "from": "13B", "to": "130B", "method": "Kronecker Width Expansion"},
            {"phase": 2, "from": "130B", "to": "1T", "method": "Depth Injection"},
            {"phase": 3, "from": "1T", "to": "10T", "method": "Cross-Domain Federation"},
            {"phase": 4, "from": "10T", "to": "1Q", "method": "Maximum Information Curriculum"}
        ],
        "description": "KRONOS scaling pipeline — 4-phase expansion from CIFN base to full-spectrum causal model"
    }

@router.get("/entity/architecture")
async def get_entity_architecture():
    """Return a full four-layer architecture report: DRSN, KRONOS, CSIE-Sheaf, and APEX-Causal."""
    if not isinstance(entity_ai, LiveEntity):
        return {"available": False, "reason": "ENTITY_MODE=mock"}
    return entity_ai.get_full_architecture_report()

@router.get("/axiom/analysis")
async def get_axiom_analysis():
    """Run the AXIOM zero-loss compression analyser over the KRONOS model weights."""
    if not isinstance(entity_ai, LiveEntity):
        return {"available": False}
    kronos_model = getattr(getattr(entity_ai, 'model', None), 'kronos', None)
    if kronos_model is None:
        return {"available": False, "reason": "kronos model not initialised"}
    return analyse_kronos_model(kronos_model)


@router.post("/kronos/scale")
async def trigger_kronos_scaling():
    """Trigger one generation of the KRONOS Godel evolutionary scaling loop."""
    global _godel_loop, _godel_results
    if not isinstance(entity_ai, LiveEntity):
        return {"available": False, "reason": "ENTITY_MODE=mock"}

    kronos_model = getattr(getattr(entity_ai, 'model', None), 'kronos', None)
    if kronos_model is None:
        return {"available": False, "reason": "kronos not initialised"}

    loop = asyncio.get_event_loop()

    base_config = {
        "vocab_size": 256,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "max_seq_len": 32,
        "memory_size": 64,
        "z_dim": 64,
        "n_slots": 4,
        "n_wave_freqs": 16,
        "dropout": 0.1,
        "kl_weight": 0.05,
        "notears_weight": 0.01,
        "notears_coeff": 0.01,
    }

    def run_one_generation():
        global _godel_loop
        if _godel_loop is None:
            _godel_loop = GodelLoop(
                base_config=base_config,
                vocab_size=256,
                population_size=3,
                n_generations=1,
                device='cpu'
            )
        result = _godel_loop.step_generation()
        _godel_results.append(result)
        return result

    result = await loop.run_in_executor(None, run_one_generation)

    global _best_evolved_config
    if result.get("best_config"):
        _best_evolved_config = result["best_config"]
        entity_ai.stats["evolved_config"] = _best_evolved_config
        entity_ai.stats["evolved_generation"] = result.get("generation", 0)
        entity_ai.stats["evolved_fitness"] = result.get("best_fitness", 0.0)

    return {
        "status": "success",
        "generation": result.get("generation", 0),
        "best_fitness": result.get("best_fitness", 0.0),
        "best_config": result.get("best_config", {}),
        "fitness_history": result.get("fitness_history", []),
        "fitness_trend": _godel_loop.fitness_trend() if _godel_loop else "",
        "total_generations_run": len(_godel_results),
        "best_evolved_config": _best_evolved_config
    }


@router.get("/kronos/scale/status")
async def get_scaling_status():
    """Return current Godel Loop scaling state and fitness history."""
    return {
        "godel_loop_active": _godel_loop is not None,
        "generations_completed": len(_godel_results),
        "fitness_history": [r.get("best_fitness", 0.0) for r in _godel_results],
        "fitness_trend": _godel_loop.fitness_trend() if _godel_loop else "",
        "latest_best_config": _godel_results[-1].get("best_config", {}) if _godel_results else {}
    }


@router.post("/axiom/compress")
async def run_axiom_compression():
    """Run the AXIOM in-place gauge fixing compression pipeline over the live KRONOS model."""
    if not isinstance(entity_ai, LiveEntity):
        return {"available": False}
    kronos_model = getattr(getattr(entity_ai, 'model', None), 'kronos', None)
    if kronos_model is None:
        return {"available": False, "reason": "kronos not initialised"}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, compress_kronos_model, kronos_model)
    return result


@router.get("/godel/auto/status")
async def get_godel_auto_status():
    """Return the current Godel Loop auto-scheduling state and fitness metrics."""
    return {
        "auto_trigger_every": 50,
        "optimize_calls_total": _optimize_call_counter,
        "next_trigger_in": 50 - (_optimize_call_counter % 50),
        "generations_completed": len(_godel_results),
        "fitness_trend": _godel_loop.fitness_trend() if _godel_loop else "not_started",
        "latest_fitness": _godel_results[-1].get("best_fitness", 0.0) if _godel_results else 0.0
    }


@router.get("/godel/best-config")
async def get_godel_best_config():
    return {
        "available": bool(_best_evolved_config),
        "best_config": _best_evolved_config,
        "generation": entity_ai.stats.get("evolved_generation", 0),
        "fitness": entity_ai.stats.get("evolved_fitness", 0.0)
    }