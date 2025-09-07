from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

# Import experiment modules
from ..experiments.ab_testing import ABTestingFramework
from ..experiments.bandits import MultiArmedBanditOptimizer
from ..experiments.sequential_testing import SequentialTestingFramework
from ..statistics.power_analysis import PowerAnalysis

app = FastAPI(
    title="Experimentation Platform API",
    description="API for A/B testing, bandits, and causal inference",
    version="1.0.0"
)

# Pydantic models
class ExperimentRequest(BaseModel):
    name: str
    metric_type: str
    variants: List[str]
    allocation: List[float]
    hypothesis: str
    effect_size: Optional[float] = 0.05
    alpha: Optional[float] = 0.05
    power: Optional[float] = 0.8

class MetricObservation(BaseModel):
    experiment_id: str
    user_id: str
    variant: str
    value: float

class BanditRequest(BaseModel):
    bandit_id: str
    arms: List[str]
    algorithm: str = 'thompson_sampling'

# Global instances
ab_framework = None
bandit_optimizer = None
sequential_tester = None
power_analyzer = None

@app.on_event("startup")
async def initialize_frameworks():
    global ab_framework, bandit_optimizer, sequential_tester, power_analyzer
    
    config = {
        'baseline_conversion_rate': 0.1,
        'baseline_std': 1.0,
        'interim_frequency': 100,
        'prior_mean': 0,
        'prior_precision': 1,
        'epsilon': 0.1,
        'ucb_c': 2.0
    }
    
    ab_framework = ABTestingFramework(config)
    bandit_optimizer = MultiArmedBanditOptimizer(config)
    sequential_tester = SequentialTestingFramework(config)
    power_analyzer = PowerAnalysis(config)
    
    logging.info("Experimentation frameworks initialized")

@app.get("/")
async def root():
    return {"message": "Experimentation Platform API", "version": "1.0.0"}

@app.post("/experiments/create")
async def create_experiment(request: ExperimentRequest):
    """Create a new A/B test experiment"""
    try:
        experiment_id = ab_framework.create_experiment(
            name=request.name,
            metric_type=request.metric_type,
            variants=request.variants,
            allocation=request.allocation,
            hypothesis=request.hypothesis,
            effect_size=request.effect_size,
            alpha=request.alpha,
            power=request.power
        )
        
        return {"experiment_id": experiment_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment"""
    try:
        success = ab_framework.start_experiment(experiment_id)
        return {"experiment_id": experiment_id, "status": "running" if success else "failed"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments/{experiment_id}/assign/{user_id}")
async def assign_variant(experiment_id: str, user_id: str):
    """Assign a user to a variant"""
    try:
        variant = ab_framework.assign_variant(experiment_id, user_id)
        return {"user_id": user_id, "variant": variant}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/experiments/record_metric")
async def record_metric(observation: MetricObservation):
    """Record a metric observation"""
    try:
        success = ab_framework.record_metric(
            observation.experiment_id,
            observation.user_id,
            observation.variant,
            observation.value
        )
        
        return {"status": "recorded" if success else "failed"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments/{experiment_id}/analyze")
async def analyze_experiment(experiment_id: str, test_type: str = 'ttest'):
    """Analyze experiment results"""
    try:
        results = ab_framework.analyze_experiment(experiment_id, test_type)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/bandits/create")
async def create_bandit(request: BanditRequest):
    """Create a multi-armed bandit"""
    try:
        bandit_id = bandit_optimizer.create_bandit(
            request.bandit_id,
            request.arms,
            request.algorithm
        )
        
        return {"bandit_id": bandit_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/bandits/{bandit_id}/select")
async def select_arm(bandit_id: str):
    """Select an arm using bandit algorithm"""
    try:
        arm = bandit_optimizer.select_arm(bandit_id)
        return {"bandit_id": bandit_id, "selected_arm": arm}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/bandits/{bandit_id}/update")
async def update_bandit_reward(bandit_id: str, arm: str, reward: float):
    """Update bandit with observed reward"""
    try:
        success = bandit_optimizer.update_reward(bandit_id, arm, reward)
        return {"status": "updated" if success else "failed"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/bandits/{bandit_id}/statistics")
async def get_bandit_stats(bandit_id: str):
    """Get bandit performance statistics"""
    try:
        stats = bandit_optimizer.get_bandit_statistics(bandit_id)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
