from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from inference import KasparovEngine

app = FastAPI(
    title="TorchFish Kasparov Hybrid API",
    description="Policy Network di Kasparov + Ricerca Tattica Quiescente"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza il motore all'avvio del server
engine = KasparovEngine(
    model_path="weights/kasparov_policy.pth",
    dict_path="models/uci_dictionary.json",
    book_csv_path="data/gm_kasparov/kasparov_moves.csv"
)


class EvaluateRequest(BaseModel):
    fen: str
    top_k: Optional[int] = 3
    search_depth: Optional[int] = 2


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "device": str(engine.device)}


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    try:
        result = engine.evaluate_position(
            fen=req.fen,
            top_k=req.top_k,
            search_depth=req.search_depth
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)