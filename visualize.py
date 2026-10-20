import chess
import chess.svg
import os
from inference import ChessEngine

def export_visual_evaluation(fen: str, model_path: str, output_path: str = "evaluation.html"):
    # Inizializza il motore caricando i pesi addestrati
    engine = ChessEngine(model_path)
    board = chess.Board(fen)
    
    # Calcola la valutazione e trova la mossa migliore (depth-1)
    norm_score, centipawns = engine.evaluate_fen(fen)
    best_move, move_score = engine.get_best_move(fen)
    
    # Determina il colore del testo in base a chi è in vantaggio
    eval_color = "#4CAF50" if centipawns > 0 else "#F44336"
    
    # Genera la scacchiera in SVG (stile Lichess) evidenziando la mossa suggerita
    board_svg = chess.svg.board(
        board=board,
        arrows=[chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#00308F80")],
        size=500
    )
    
    # Struttura HTML con CSS minimale
    html = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>TorchFish - Analisi Posizione</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #262421; color: #dbd7d2; display: flex; flex-direction: column; align-items: center; padding: 2rem; margin: 0; }}
            .eval-box {{ background: #302e2b; padding: 1.5rem 2.5rem; border-radius: 8px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #403d39; }}
            .eval-box h2 {{ margin: 0 0 10px 0; font-size: 1.2rem; color: #a9a9a9; text-transform: uppercase; letter-spacing: 1px; }}
            .score {{ font-size: 2.5rem; font-weight: bold; margin: 10px 0; color: {eval_color}; }}
            .details {{ font-size: 1.1rem; margin: 5px 0; }}
            .suggested-move {{ font-size: 1.2rem; margin-top: 15px; padding-top: 15px; border-top: 1px solid #403d39; }}
            svg {{ border-radius: 4px; box-shadow: 0 10px 15px rgba(0,0,0,0.5); }}
        </style>
    </head>
    <body>
        <div class="eval-box">
            <h2>TorchFish Engine</h2>
            <div class="score">{centipawns:+.1f} cp</div>
            <div class="details">Score di Rete: {norm_score:+.3f}</div>
            <div class="suggested-move">Mossa Consigliata: <b>{board.san(best_move)}</b></div>
        </div>
        {board_svg}
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Analisi esportata con successo. Apri il file nel browser: file://{os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Esempio: Difesa Siciliana
    test_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    # Richiede che best_chess_model.pth esista nella stessa cartella
    export_visual_evaluation(test_fen, "best_chess_model.pth")