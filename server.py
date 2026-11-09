import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from inference import KasparovEngine

app = FastAPI(
    title="TorchFish Legends Hybrid API",
    description="Motore Ibrido Alpha-Beta PeSTO + Policy Network Multi-Leggenda")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = KasparovEngine(
    model_path="weights/kasparov_policy.pth",
    dict_path="models/uci_dictionary.json",
    book_csv_path="data/gm_kasparov/kasparov_moves.csv"
)


class EvaluateRequest(BaseModel):
    fen: str = Field(..., min_length=15, max_length=120)
    legend: Optional[str] = "kasparov"
    top_k: int = Field(default=3, ge=1, le=10)
    history_fens: Optional[List[str]] = None


@app.get("/api/legends")
def get_legends():
    return {"legends": engine.get_legends_info()}


@app.post("/api/eval_only")
def eval_only(req: EvaluateRequest):
    try:
        chess.Board(req.fen)
        return engine.evaluate_only(fen=req.fen, legend=req.legend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        chess.Board(req.fen)
        return engine.evaluate_position(
            fen=req.fen,
            top_k=req.top_k,
            legend=req.legend,
            history_fens=req.history_fens
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)