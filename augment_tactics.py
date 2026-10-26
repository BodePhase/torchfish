import csv
import json
import random
import chess
from tqdm import tqdm

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def find_tactical_punishment(board: chess.Board, valid_uci_set: set):
    """
    Cerca nella posizione attuale:
    1. Uno Scacco Matto in 1 mossa.
    2. La cattura del pezzo avversario di maggior valore lasciato indifeso.
    """
    legal_moves = list(board.legal_moves)

    # 1. Priorità assoluta: Scacco Matto in 1
    for move in legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate and move.uci() in valid_uci_set:
            return move.uci(), "mate_in_1"

    # 2. Cattura di pezzi in presa (guadagno netto elevato)
    best_capture = None
    best_gain = 0

    for move in legal_moves:
        if not board.is_capture(move) or move.uci() not in valid_uci_set:
            continue

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if not victim or not attacker:
            continue

        victim_val = PIECE_VALUES.get(victim.piece_type, 0)
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0)

        board.push(move)
        is_recaptured = board.is_attacked_by(board.turn, move.to_square)
        board.pop()

        # Se la casa non è difesa dall'avversario, il guadagno è l'intero pezzo;
        # altrimenti è la differenza tra pezzo catturato e pezzo catturante
        net_gain = victim_val if not is_recaptured else (victim_val - attacker_val)

        if net_gain >= 3 and net_gain > best_gain:
            best_gain = net_gain
            best_capture = move.uci()

    if best_capture:
        return best_capture, "hanging_piece"

    return None, None


def generate_tactical_dataset(
    input_csv="data/gm_kasparov/kasparov_moves.csv",
    dict_path="models/uci_dictionary.json",
    output_csv="data/gm_kasparov/kasparov_tactical_augmented.csv",
    max_synthetic_samples=25000
):
    # --- CARICAMENTO UNIVERSALE E RICORSIVO DEL DIZIONARIO UCI ---
    with open(dict_path, "r", encoding="utf-8") as f:
        raw_dict = json.load(f)

    valid_uci = set()

    def extract_moves(data):
        if isinstance(data, dict):
            for k, v in data.items():
                # Se la chiave è una stringa lunga 4 o 5 caratteri che inizia per a-h e finisce per 1-8
                if isinstance(k, str) and len(k) in [4, 5] and k[0] in "abcdefgh" and k[1] in "12345678":
                    valid_uci.add(k)
                # O se il valore è la mossa
                elif isinstance(v, str) and len(v) in [4, 5] and v[0] in "abcdefgh" and v[1] in "12345678":
                    valid_uci.add(v)
                
                # Continua a scendere nei sotto-dizionari
                if isinstance(v, (dict, list)):
                    extract_moves(v)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and len(item) in [4, 5] and item[0] in "abcdefgh" and item[1] in "12345678":
                    valid_uci.add(item)
                elif isinstance(item, (dict, list)):
                    extract_moves(item)

    extract_moves(raw_dict)
    print(f"[Augment] Trovate e caricate {len(valid_uci)} mosse valide dal dizionario UCI.")
    # -------------------------------------------------------------

    with open(input_csv, "r", encoding="utf-8") as f:
        original_rows = list(csv.DictReader(f))

    print(f"[Augment] Letti {len(original_rows)} esempi storici da {input_csv}.")

    tactical_rows = []
    stats = {"mate_in_1": 0, "hanging_piece": 0}

    random.shuffle(original_rows)

    for row in tqdm(original_rows, desc="Generazione tattiche e punizione blunder"):
        if len(tactical_rows) >= max_synthetic_samples:
            break

        try:
            board = chess.Board(row["fen"])
        except Exception:
            continue

        # Caso A: La posizione originale ha già un matto in 1 o un pezzo gratis
        uci_punish, kind = find_tactical_punishment(board, valid_uci)
        if uci_punish:
            tactical_rows.append({"fen": board.fen(), "move_uci": uci_punish})
            stats[kind] += 1
            continue

        # Caso B: Simula una mossa "blunder" dell'avversario che mette un pezzo in presa
        legal_moves = [m for m in board.legal_moves if board.piece_type_at(m.from_square) in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)]
        random.shuffle(legal_moves)

        for blunder_candidate in legal_moves[:4]:
            board.push(blunder_candidate)
            if not board.is_game_over():
                punish_move, kind = find_tactical_punishment(board, valid_uci)
                if punish_move:
                    tactical_rows.append({"fen": board.fen(), "move_uci": punish_move})
                    stats[kind] += 1
                    board.pop()
                    break
            board.pop()

    # Unisce il dataset storico con il nuovo blocco tattico
    combined_rows = original_rows + tactical_rows
    random.shuffle(combined_rows)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci"])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"\n[Completato] Salvate {len(combined_rows)} posizioni totali in {output_csv}")
    print(f" -> Matti in 1 generati/insegnati: {stats['mate_in_1']}")
    print(f" -> Catture di pezzi in presa generate/insegnate: {stats['hanging_piece']}")


if __name__ == "__main__":
    generate_tactical_dataset()