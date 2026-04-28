import uuid
import threading
from fastapi import APIRouter, BackgroundTasks
from src.api.schemas import SimulateRequest, JobStatusResponse
from src.simulation.monte_carlo import run_monte_carlo

router = APIRouter()

# Simple in-memory job store
jobs = {}

def run_simulation_task(job_id: str, n_simulations: int):
    jobs[job_id]["status"] = "running"
    try:
        run_monte_carlo(n_simulations=n_simulations)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@router.post("", response_model=JobStatusResponse)
def trigger_simulation(request: SimulateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}
    
    background_tasks.add_task(run_simulation_task, job_id, request.n_simulations)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "progress": 0
    }

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_simulation_status(job_id: str):
    if job_id not in jobs:
        return {"job_id": job_id, "status": "not_found", "progress": 0}
        
    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "progress": jobs[job_id].get("progress", 0)
    }
