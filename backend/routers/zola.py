import logging
from fastapi import APIRouter, HTTPException
from core.entity_resolution import entity_registry
from entity_interface.mock_entity import MockEntity
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

# KRONOSOrchestrator remains available as a library import in
# entity_interface/kronos/orchestrator.py for future use,
# but is not currently wired into any live endpoint.

router = APIRouter(prefix="/api/zola", tags=["zola"])

# Always import LiveEntity class so isinstance() checks work in all modes
# (Instantiation only happens when ENTITY_MODE=live)
from entity_interface.live_entity import LiveEntity

# Dynamically instantiate the entity layer based on config
if ENTITY_MODE == "live":
    entity_ai = LiveEntity()
else:
    entity_ai = MockEntity()

# Godel Loop state — persists across requests within a single server process
_godel_loop = None
_godel_results = []
_best_evolved_config = {}

logger = logging.getLogger(__name__)

# Optimize-step counter for auto-triggering Godel generations
_optimize_call_counter = 0

async def save_prediction_to_db(prediction: dict) -> bool:
    """Persist a prediction dict to the DB. Returns True on success, False on failure.

    Fields marked 'untrained_heuristic' in the prediction (optimal_intervention,
    recommended_timing, success_probability) are stored with a sentinel value of -1.0
    for success_probability so the nullable=False DB constraint is not violated, while
    making it clear in the persisted record that these are not learned predictions.
    """
    entity_id = prediction.get("entity_id", "<unknown>")
    try:
        async with async_session_maker() as session:
            prediction_type = prediction.get("type", "behavioral")
            # confidence is now from the trained transition head (softmax margin)
            confidence = prediction.get("confidence", 0.0) or 0.0
            predicted_outcome = prediction.get("prediction", "")
            causal_factors = json.dumps(prediction.get("causal_factors", []))
            is_heuristic = prediction.get("untrained_heuristic", False)

            # success_probability=None signals untrained head — store -1.0 as sentinel
            raw_sp = prediction.get("success_probability")
            stored_success_prob = float(raw_sp) if (raw_sp is not None) else -1.0

            row = PredictionModel(
                entity_id=entity_id,
                transition_type=prediction.get("transition_type", prediction_type),
                causal_mechanism=prediction.get("causal_mechanism", predicted_outcome),
                # optimal_intervention and recommended_timing may be heuristic; stored as-is
                # but caller is informed via the untrained_heuristic flag
                optimal_intervention=prediction.get("optimal_intervention", "[heuristic]" if is_heuristic else ""),
                success_probability=stored_success_prob,
                recommended_timing=prediction.get("recommended_timing", "[heuristic]" if is_heuristic else ""),
                consequence_chain=prediction.get("consequence_chain", json.loads(causal_factors))
            )
            session.add(row)
            await session.commit()
            return True
    except Exception as e:
        logger.error(
            "[ZOLA] DB prediction save failed for entity_id=%s: %s",
            entity_id, e, exc_info=True
        )
        return False

@router.get("/predictions")
async def get_predictions():
    entities = [e for e in entity_registry.get_all() if e["status"] == "pre-transition"][:8]
    predictions = []

    for entity in entities:
        # Pass entropy explicitly in context
        prediction = await entity_ai.predict(entity["id"], {"entropy": entity["entropy"]})
        prediction["entity_name"] = entity["name"]
        prediction["domain"] = entity["domain"]
        # Persist and surface failure without blocking the prediction result
        persisted = await save_prediction_to_db(prediction)
        prediction["persisted"] = persisted
        predictions.append(prediction)
    
    return predictions

@router.get("/dashboard")
async def get_zola_dashboard():
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from database import async_session_maker
    from models.commerce import CompanyModel
    from services.narrative_engine import NarrativeEngine

    # Sector → transition type mapping for real semantic labeling
    SECTOR_TRANSITIONS = {
        "Technology": "market_expansion",
        "Healthcare": "regulatory_pivot",
        "Finance": "capital_reallocation",
        "Energy": "infrastructure_scaling",
        "Consumer": "demand_surge",
        "Industrials": "supply_chain_shift",
        "Materials": "commodity_cycle",
        "Utilities": "regulatory_pivot",
        "Real Estate": "capital_reallocation",
        "Communication": "market_expansion",
    }
    SECTOR_INTERVENTIONS = {
        "Technology": "Accelerate R&D headcount and cloud infrastructure investment",
        "Healthcare": "Initiate regulatory pre-submission engagement and clinical pipeline review",
        "Finance": "Rebalance capital allocation toward high-yield credit instruments",
        "Energy": "Fast-track permitting for capacity expansion projects",
        "Consumer": "Expand distribution channels and geographic market penetration",
        "Industrials": "Diversify supplier base and pre-build strategic inventory",
        "Materials": "Hedge commodity price exposure via forward contracts",
        "Utilities": "Lobby for rate case approval and accelerate grid modernization capex",
        "Real Estate": "Refinance maturing debt and lock in long-term fixed rates",
        "Communication": "Accelerate subscriber acquisition and content licensing deals",
    }

    try:
        async with async_session_maker() as session:
            comp_res = await session.execute(
                select(CompanyModel)
                .options(
                    selectinload(CompanyModel.financial_metrics),
                    selectinload(CompanyModel.job_postings)
                )
                .limit(10)
            )
            companies = comp_res.scalars().all()

        dashboard_predictions = []
        for company in companies:
            jobs_count = len(company.job_postings)
            sec_count = len(company.financial_metrics)
            # Revenue from first financial metric record (if available)
            revenue_val = getattr(company.financial_metrics[0], 'revenue', None) if sec_count > 0 else None
            revenue_b = round((revenue_val or 0) / 1e9, 2)
            news_sent = getattr(company, 'news_sentiment', 0.0) or 0.0

            # Weighted expansion score from real signals
            score = round(
                0.4 * min(jobs_count / 10.0, 1.0) +
                0.3 * min(sec_count / 5.0, 1.0) +
                0.2 * min(revenue_b / 100.0, 1.0) +
                0.1 * news_sent,
                4
            )

            val = (hash(company.id) % 100) / 100.0
            current_entropy = round(0.3 + val * 0.4, 4)

            sector = company.sector or "Technology"
            transition_type = SECTOR_TRANSITIONS.get(sector, "behavioral_shift")
            optimal_intervention = SECTOR_INTERVENTIONS.get(sector, "Monitor closely and prepare contingency capital")

            # Consequence chain built from real signals
            consequence_chain = []
            if jobs_count > 5:
                consequence_chain.append(f"Headcount velocity +{jobs_count} roles")
            if sec_count > 0:
                consequence_chain.append(f"{sec_count} SEC filing signals")
            if revenue_b > 0:
                consequence_chain.append(f"Revenue base ${revenue_b}B")
            consequence_chain.append("→ Pre-transition behavior detected")

            narrative = await NarrativeEngine.generate_short_summary(company.ticker, company.legal_name, score)

            prediction = {
                "entity_id": company.id,
                "transition_type": transition_type,
                "confidence": round(score, 4),
                "causal_mechanism": f"Signal convergence: {jobs_count} job postings + {sec_count} SEC filings + ${revenue_b}B revenue base in {sector}",
                "optimal_intervention": optimal_intervention,
                "recommended_timing": "Q3 2025" if score > 0.5 else "Q4 2025",
                "consequence_chain": consequence_chain,
            }

            dashboard_predictions.append({
                "company_id": company.id,
                "ticker": company.ticker,
                "legal_name": company.legal_name,
                "domain": sector,
                "expansion_score": score,
                "current_entropy": current_entropy,
                "narrative": narrative,
                "prediction_details": prediction
            })

        dashboard_predictions.sort(key=lambda x: x["expansion_score"], reverse=True)

        return {
            "predictions": dashboard_predictions[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
        # Mock mode has no model — virtual_parameters is null (not a real measurement).
        # virtual_parameters_disclosed: False signals to callers that this field
        # does not trace to a sum(p.numel()) call on any real model.
        return {
            "entity_mode": "mock",
            "stats": {
                "virtual_parameters": None,
                "virtual_parameters_disclosed": False,
                "virtual_parameters_note": (
                    "No model is active in mock mode. "
                    "Set ENTITY_MODE=live to get a real parameter count."
                ),
                "wave_basis_size_kb": 0.0,
                "backprop_steps": 0,
                "latest_loss": 0.0,
                "latest_grad_norm": 0.0,
                "facts_crawled": 0,
                "self_evolution_cycles": 0,
                "pending_patches": [],
                "approved_patches": []
            },
            "actual_stored_params": None,
            "actual_stored_params_note": "No model active in mock mode.",
            "wave_basis_size_kb": 0.0,
            "architecture_summary": {
                "representation": "Mock mode — MockEntity returns random strings, no computation",
                "layer1": "N/A (mock mode)",
                "layer2": "N/A (mock mode)",
                "storage_model": "N/A (mock mode)"
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
    drsn_result = entity_ai.drsn.encode_features(features.squeeze().tolist(), n_steps=10)
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
        "drsn_active_nodes": drsn_result.get("active_nodes", 0),
        "drsn_total_spikes": drsn_result.get("total_spikes", 0),
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
        "available": False,
        "current_phase": None,
        "phase_label": "not started — see phases list for planned theoretical stages",
        "phases": [
            {"phase": 1, "from": "13B (Theoretical)", "to": "130B (Theoretical)", "method": "Kronecker Width Expansion"},
            {"phase": 2, "from": "130B (Theoretical)", "to": "1T (Theoretical)", "method": "Depth Injection"},
            {"phase": 3, "from": "1T (Theoretical)", "to": "10T (Theoretical)", "method": "Cross-Domain Federation"},
            {"phase": 4, "from": "10T (Theoretical)", "to": "1Q (Theoretical)", "method": "Maximum Information Curriculum"}
        ],
        "description": "KRONOS theoretical scaling roadmap — not currently implemented, active, or in progress",
        "scaling_pipeline_active": False,
        "scaling_pipeline_note": "implemented as dead/experimental code but not active in current runtime",
        "orchestrator_status": "library_code_only_not_instantiated_in_live_path"
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
        "best_evolved_config": _best_evolved_config,
        "fitness_type": "structural_topology_only_no_task_data",
    }


@router.get("/kronos/scale/status")
async def get_scaling_status():
    """Return current Godel Loop scaling state and fitness history."""
    return {
        "godel_loop_active": _godel_loop is not None,
        "generations_completed": len(_godel_results),
        "fitness_history": [r.get("best_fitness", 0.0) for r in _godel_results],
        "fitness_trend": _godel_loop.fitness_trend() if _godel_loop else "",
        "latest_best_config": _godel_results[-1].get("best_config", {}) if _godel_results else {},
        "fitness_type": "structural_topology_only_no_task_data",
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
        "latest_fitness": _godel_results[-1].get("best_fitness", 0.0) if _godel_results else 0.0,
        "fitness_type": "structural_topology_only_no_task_data",
    }


@router.get("/godel/best-config")
async def get_godel_best_config():
    return {
        "available": bool(_best_evolved_config),
        "best_config": _best_evolved_config,
        "generation": entity_ai.stats.get("evolved_generation", 0),
        "fitness": entity_ai.stats.get("evolved_fitness", 0.0),
        "fitness_type": "structural_topology_only_no_task_data",
    }