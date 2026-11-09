import os
import sys
import warnings

# Aggiunge la cartella root di TorchFish al path di Python per poter importare i file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
import torch
from fastapi.testclient import TestClient

# Ignora i fastidiosi warning interni di FastAPI/Starlette durante i test
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dataset_policy import board_to_tensor
from server import app, engine

client = TestClient(app)


def test_board_to_tensor_orientation_and_shape():
    """Verifica che il tensore sia 14x8x8 e che la Traversa 1 sia mappata sulla riga 0 (R1)."""
    board = chess.Board()
    tensor = board_to_tensor(board)

    assert tensor.shape == (14, 8, 8)
    assert tensor.dtype == torch.float32
    # Pedone Bianco in e2 (rank=1, file=4) -> canale 0, riga 1, colonna 4
    assert tensor[0, 1, 4].item() == 1.0
    # Re Bianco in e1 (rank=0, file=4) -> canale 5, riga 0, colonna 4
    assert tensor[5, 0, 4].item() == 1.0
    # Re Nero in e8 (rank=7, file=4) -> canale 11 (5+6), riga 7, colonna 4
    assert tensor[11, 7, 4].item() == 1.0
    # Turno del Bianco -> canale 12 a 1.0
    assert tensor[12, 0, 0].item() == 1.0


def test_mate_in_one_execution():
    """Verifica che il motore trovi sempre uno Scacco Matto in 1 mossa."""
    # Matto del barbiere pronto: Qxf7#
    scholar_fen = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    res = engine.evaluate_position(scholar_fen, top_k=3, legend="kasparov")

    assert res["best_move_uci"] == "h5f7"
    assert res["mate_in"] == 1


def test_free_queen_capture_blunder_lock():
    """Verifica che il motore catturi immediatamente una Donna avversaria lasciata indifesa."""
    # Donna Nera indifesa in d5, attaccata dal Cavallo Bianco in c3
    free_queen_fen = "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 2"
    res = engine.evaluate_position(free_queen_fen, top_k=3, legend="kasparov")

    assert res["best_move_uci"] == "c3d5"


def test_fastapi_endpoints_and_validation():
    """Verifica le risposte di /api/legends, /api/eval_only, /api/evaluate e i vincoli Pydantic (R3)."""
    r_leg = client.get("/api/legends")
    assert r_leg.status_code == 200
    assert len(r_leg.json()["legends"]) == 4

    r_eval = client.post("/api/eval_only", json={"fen": chess.STARTING_FEN, "legend": "kasparov"})
    assert r_eval.status_code == 200
    assert "eval_white" in r_eval.json()

    # Verifica rifiuto top_k=0 (HTTP 422 Unprocessable Entity)
    r_invalid = client.post("/api/evaluate", json={"fen": chess.STARTING_FEN, "top_k": 0})
    assert r_invalid.status_code == 422