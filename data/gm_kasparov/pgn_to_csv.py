import chess.pgn
import pandas as pd
import os

def extract_kasparov_moves(pgn_file_path, output_csv_path):
    # Lista che conterrà i dizionari con FEN e la mossa UCI
    dataset = []
    games_processed = 0

    print(f"Apertura del file {pgn_file_path}...")
    
    # Molti file PGN vecchi usano latin-1 o utf-8. Ignoriamo gli errori di decodifica dei nomi strani.
    with open(pgn_file_path, "r", encoding="utf-8", errors="replace") as pgn:
        while True:
            # Legge la partita successiva
            game = chess.pgn.read_game(pgn)
            
            # Se restituisce None, siamo arrivati alla fine del file
            if game is None:
                break
                
            # Identifichiamo i giocatori per capire chi è Kasparov
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            
            is_white = "Kasparov" in white_player
            is_black = "Kasparov" in black_player
            
            # Se Kasparov non è in questa partita (strano in un suo DB, ma meglio essere sicuri), salta
            if not is_white and not is_black:
                continue

            # Inizializziamo la scacchiera alla posizione di partenza
            board = game.board()

            # Scorriamo tutte le mosse giocate nella partita
            for move in game.mainline_moves():
                # Vogliamo clonare solo il comportamento di Kasparov, non del suo avversario!
                # Salviamo la posizione SOLO SE è il turno di Kasparov
                if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and is_black):
                    dataset.append({
                        "fen": board.fen(),
                        "move_uci": move.uci()
                    })
                
                # Eseguiamo la mossa sulla scacchiera per passare al turno successivo
                board.push(move)

            games_processed += 1
            if games_processed % 500 == 0:
                print(f"Elaborate {games_processed} partite...")

    print(f"\nEstrazione completata! Trovate {len(dataset)} mosse uniche giocate da Kasparov.")
    
    # Conversione in DataFrame e salvataggio
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset salvato con successo in: {output_csv_path}")

if __name__ == "__main__":
    # Assicurati di avere il file Kasparov.pgn nella stessa cartella
    INPUT_PGN = "Kasparov.pgn"
    OUTPUT_CSV = "kasparov_moves.csv"
    
    extract_kasparov_moves(INPUT_PGN, OUTPUT_CSV)