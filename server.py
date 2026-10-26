from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from inference import KasparovEngine

app = FastAPI(
    title="TorchFish Legends AlphaZero-MCTS API",
    description="Motore MCTS PUCT Multi-Leggenda (Kasparov, Tal, Karpov, Carlsen)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = KasparovEngine(
    model_path="weights/kasparov_policy.pth",
    value_model_path="best_chess_model.pth",
    dict_path="models/uci_dictionary.json",
    book_csv_path="data/gm_kasparov/kasparov_moves.csv"
)


class EvaluateRequest(BaseModel):
    fen: str
    legend: Optional[str] = "kasparov"
    top_k: Optional[int] = 3
    num_simulations: Optional[int] = 64


@app.get("/api/legends")
async def get_legends():
    return {"legends": engine.get_legends_info()}


@app.post("/api/eval_only")
async def eval_only(req: EvaluateRequest):
    try:
        return engine.evaluate_only(fen=req.fen, legend=req.legend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    try:
        return engine.evaluate_position(
            fen=req.fen,
            top_k=req.top_k,
            num_simulations=req.num_simulations,
            legend=req.legend
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)