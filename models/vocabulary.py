import chess
import json

def generate_uci_vocabulary():
    uci_moves = []
    
    # Scorriamo tutte le combinazioni possibili tra le 64 case
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq != to_sq:
                # Mossa base (es. 'e2e4' o 'a1h8')
                base_move = chess.square_name(from_sq) + chess.square_name(to_sq)
                uci_moves.append(base_move) # Copre i movimenti di tutti i pezzi normali
                
                # Promozioni: si verificano solo dalla traversa 7 alla 8 (Bianco) o 2 alla 1 (Nero)
                is_white_promo = (base_move[1] == '7' and base_move[3] == '8')
                is_black_promo = (base_move[1] == '2' and base_move[3] == '1')
                
                if is_white_promo or is_black_promo:
                    # Se lo spostamento tocca l'ultima traversa, aggiungiamo le 4 promozioni
                    for promo in ['q', 'r', 'b', 'n']:
                        uci_moves.append(base_move + promo)
                        
    # Rimuoviamo duplicati e ordiniamo alfabeticamente per avere indici stabili
    uci_moves = sorted(list(set(uci_moves)))
    
    # Creiamo i due dizionari: Da Stringa a Indice (per l'addestramento) e viceversa (per l'inferenza)
    move_to_int = {move: i for i, move in enumerate(uci_moves)}
    int_to_move = {i: move for i, move in enumerate(uci_moves)}
    
    return move_to_int, int_to_move

if __name__ == "__main__":
    m2i, i2m = generate_uci_vocabulary()
    
    # Il numero totale sarà 4544 (4032 mosse standard + 512 combinazioni di promozione)
    print(f"Vocabolario generato con {len(m2i)} mosse totali possibili.")
    
    # Salviamo i dizionari in un file JSON per poterli caricare velocemente dalla rete neurale
    with open("uci_dictionary.json", "w") as f:
        json.dump({"move_to_int": m2i, "int_to_move": i2m}, f)
        
    print("Dizionario salvato in 'uci_dictionary.json'.")