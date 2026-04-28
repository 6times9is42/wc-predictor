from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import teams, predictions, simulate

app = FastAPI(
    title="WC 2026 Predictor API",
    description="ML-powered FIFA 2026 World Cup win probability predictions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams.router, prefix="/teams", tags=["teams"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(simulate.router, prefix="/simulate", tags=["simulate"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the WC 2026 Predictor API"}
