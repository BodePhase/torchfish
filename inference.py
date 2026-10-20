import os
import csv
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
import chess

from models.policy_net import KasparovPolicyNet

# Valori dei pezzi in centesimi di pedone (Centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Tabelle posizionali (Piece-Square Tables) dal punto di vista del Bianco
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}


class KasparovEngine:
    def __init__(
        self,
        model_path: str = "weights/kasparov_policy.pth",
        dict_path: str = "models/uci_dictionary.json",
        book_csv_path: str = "data/gm_kasparov/kasparov_moves.csv"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Carica il dizionario UCI (4544 mosse)
        with open(dict_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)

        # Gestisce sia il formato {"e2e4": 0} sia una lista ["e2e4", ...]
        if isinstance(raw_dict, dict):
            if all(isinstance(v, int) for v in raw_dict.values()):
                self.move_to_idx = raw_dict
                self.idx_to_move = {v: k for k, v in raw_dict.items()}
            else:
                self.idx_to_move = {int(k): v for k, v in raw_dict.items()}
                self.move_to_idx = {v: int(k) for k, v in raw_dict.items()}
        elif isinstance(raw_dict, list):
            self.idx_to_move = {i: m for i, m in enumerate(raw_dict)}
            self.move_to_idx = {m: i for i, m in enumerate(raw_dict)}

        num_classes = len(self.move_to_idx)

        # 2. Inizializza e carica KasparovPolicyNet
        try:
            self.model = KasparovPolicyNet(num_classes=num_classes).to(self.device)
        except TypeError:
            self.model = KasparovPolicyNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Rileva automaticamente il numero di canali in ingresso del primo layer conv
        first_param = next(self.model.parameters())
        self.in_channels = first_param.shape[1] if first_param.ndim == 4 else 12

        # 3. Carica l'Opening Book storico da kasparov_moves.csv (se presente)
        self.opening_book = {}
        if os.path.exists(book_csv_path):
            try:
                with open(book_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Usa i primi 4 campi del FEN (pezzi, turno, arrocco, en-passant)
                        fen_key = " ".join(row["fen"].split()[:4])
                        uci = row["move_uci"].strip()
                        if fen_key not in self.opening_book:
                            self.opening_book[fen_key] = {}
                        self.opening_book[fen_key][uci] = self.opening_book[fen_key].get(uci, 0) + 1
                print(f"[TorchFish] Caricate {len(self.opening_book)} posizioni storiche di Kasparov.")
            except Exception as e:
                print(f"[TorchFish] Avviso caricamento book CSV: {e}")

        print(f"[TorchFish] KasparovEngine pronto su {self.device} ({self.in_channels} canali).")

    def fen_to_tensor(self, fen: str) -> torch.Tensor:
        """Converte il FEN nel tensore multi-canale atteso da KasparovPolicyNet."""
        try:
            from dataset_policy import fen_to_tensor as custom_encoder
            t = custom_encoder(fen)
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).float()
            if t.shape[0] == self.in_channels:
                return t
        except Exception:
            pass

        board = chess.Board(fen)
        tensor = np.zeros((self.in_channels, 8, 8), dtype=np.float32)

        piece_Order = [
            (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
            (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
            (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
            (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
        ]

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece_Order.index((piece.piece_type, piece.color))
                row = 7 - (square // 8)
                col = square % 8
                tensor[channel, row, col] = 1.0

        if self.in_channels >= 13:
            tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        return torch.from_numpy(tensor)

    # =========================================================================
    # MOTORE TATTICO (Valutazione Statica + Quiescence + Negamax Alpha-Beta)
    # =========================================================================
    def _static_eval(self, board: chess.Board) -> int:
        """Valuta la scacchiera in centipawn dal punto di vista di chi ha il turno."""
        if board.is_checkmate():
            return -99999
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue
            val = PIECE_VALUES[piece.piece_type]
            # Indice PST (specchiato per il Nero)
            pst_idx = (sq ^ 56) if piece.color == chess.WHITE else sq
            pos_bonus = PST[piece.piece_type][pst_idx]

            if piece.color == chess.WHITE:
                score += val + pos_bonus
            else:
                score -= val + pos_bonus

        # Penalità per pezzi in presa (attaccati da pezzi di minor valore o non difesi)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece or piece.piece_type == chess.KING:
                continue
            enemy = not piece.color
            if board.is_attacked_by(enemy, sq):
                defenders = board.attackers(piece.color, sq)
                if not defenders:
                    # Pezzo completamente indifeso!
                    hang_penalty = int(PIECE_VALUES[piece.piece_type] * 0.45)
                    score += -hang_penalty if piece.color == chess.WHITE else hang_penalty

        return score if board.turn == chess.WHITE else -score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, q_depth: int = 0) -> int:
        """Risolve tutte le catture forzate per evitare l'effetto orizzonte (pezzi in presa)."""
        stand_pat = self._static_eval(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if q_depth >= 4:
            return stand_pat

        # Ordina le catture secondo MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(
            key=lambda m: (
                PIECE_VALUES.get(board.piece_type_at(m.to_square) or chess.PAWN, 100) * 10
                - PIECE_VALUES.get(board.piece_type_at(m.from_square) or chess.PAWN, 100)
            ),
            reverse=True
        )

        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, q_depth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        """Ricerca tattica Negamax con potatura Alpha-Beta."""
        if board.is_checkmate():
            return -(99000 + depth * 100)
        if board.is_game_over():
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        moves = list(board.legal_moves)
        # Dai priorità a catture, promozioni e scacchi per velocizzare il pruning
        moves.sort(
            key=lambda m: (
                (1000 if board.is_capture(m) else 0)
                + (800 if m.promotion else 0)
                + (400 if board.gives_check(m) else 0)
            ),
            reverse=True
        )

        best_val = -999999
        for move in moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if val > best_val:
                best_val = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        return best_val

    # =========================================================================
    # VALUTAZIONE PRINCIPALE IBRIDA (Policy Net + Tattica + Book)
    # =========================================================================
    def evaluate_position(self, fen: str, top_k: int = 3, search_depth: int = 2):
        board = chess.Board(fen)
        if board.is_game_over():
            return {
                "best_move_san": None,
                "best_move_uci": None,
                "confidence": 0.0,
                "top_moves": [],
                "is_book_move": False
            }

        legal_moves = list(board.legal_moves)

        # 1. Inferenza della Rete Neurale (Stile di Kasparov)
        input_tensor = self.fen_to_tensor(fen).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor).squeeze(0)
            probs = F.softmax(logits, dim=-1)

        policy_priors = {}
        total_legal_prob = 0.0
        for move in legal_moves:
            uci = move.uci()
            p = float(probs[self.move_to_idx[uci]].item()) if uci in self.move_to_idx else 1e-6
            policy_priors[uci] = p
            total_legal_prob += p

        norm_factor = total_legal_prob if total_legal_prob > 0 else 1.0
        for uci in policy_priors:
            policy_priors[uci] /= norm_factor

        # Controlla se la posizione è nel libro storico di Kasparov
        fen_key = " ".join(fen.split()[:4])
        book_hits = self.opening_book.get(fen_key, {})
        total_book_games = sum(book_hits.values()) if book_hits else 0

        # Valutazione statica di partenza per misurare il guadagno/perdita materiale della mossa
        base_eval = self._static_eval(board)

        # Seleziona le mosse da sottoporre a verifica tattica profonda:
        # Top-10 mosse di Kasparov + TUTTE le catture, promozioni e scacchi (per non perdere mai tatticismi!)
        sorted_by_policy = sorted(legal_moves, key=lambda m: policy_priors[m.uci()], reverse=True)
        moves_to_search = set(sorted_by_policy[:10])
        for m in legal_moves:
            if board.is_capture(m) or board.gives_check(m) or m.promotion:
                moves_to_search.add(m)

        evaluated_candidates = []

        for move in legal_moves:
            uci = move.uci()
            san = board.san(move)
            prior = policy_priors[uci]

            board.push(move)

            # A. MATTO IN 1 IMMEDIATO -> Gioca istantaneamente!
            if board.is_checkmate():
                board.pop()
                return {
                    "best_move_san": san,
                    "best_move_uci": uci,
                    "confidence": 100.0,
                    "top_moves": [{"san": san, "uci": uci, "prob": 100.0}],
                    "is_book_move": False
                }

            # B. Valutazione Tattica (Depth search per le mosse principali, 1-ply Quiescence per le secondarie)
            if move in moves_to_search:
                tactical_cp = -self._negamax(board, depth=search_depth - 1, alpha=-999999, beta=999999)
            else:
                tactical_cp = -self._quiescence(board, alpha=-999999, beta=999999)

            board.pop()

            # Delta tattico rispetto alla posizione attuale (in centipawn)
            delta_cp = tactical_cp - base_eval

            # Bonus stile Kasparov: log-probabilità scalata in centipawn (max ~140 cp di influenza posizionale)
            # Così a parità tattica comanda SEMPRE Kasparov, ma se una mossa perde un pezzo (-300 cp) viene scartata!
            style_bonus_cp = 45.0 * math.log(max(prior, 1e-4))

            # Bonus Opening Book storico di Kasparov (se la mossa non è un blunder tattico)
            book_bonus_cp = 0.0
            if uci in book_hits and delta_cp > -120:
                book_freq = book_hits[uci] / total_book_games
                book_bonus_cp = 120.0 * book_freq

            composite_score = tactical_cp + style_bonus_cp + book_bonus_cp

            evaluated_candidates.append({
                "san": san,
                "uci": uci,
                "prior_prob": prior,
                "tactical_cp": tactical_cp,
                "composite_score": composite_score,
                "in_book": uci in book_hits
            })

        # Ordina per punteggio composito (Tattica + Stile Kasparov)
        evaluated_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        # Calcola le percentuali di confidenza finali tramite Softmax con temperatura sui top candidati
        top_slice = evaluated_candidates[:top_k]
        max_score = top_slice[0]["composite_score"]
        # Temperatura di 80 centipawn per distribuire le probabilità in modo leggibile nella UI
        exp_scores = [math.exp(max(-20.0, (c["composite_score"] - max_score) / 80.0)) for c in top_slice]
        sum_exp = sum(exp_scores) or 1.0

        top_moves = []
        for c, e_val in zip(top_slice, exp_scores):
            top_moves.append({
                "san": c["san"],
                "uci": c["uci"],
                "prob": round((e_val / sum_exp) * 100, 1),
                "eval_cp": round(c["tactical_cp"] / 100.0, 2)
            })

        best = top_moves[0]
        return {
            "best_move_san": best["san"],
            "best_move_uci": best["uci"],
            "confidence": best["prob"],
            "top_moves": top_moves,
            "is_book_move": top_slice[0]["in_book"]
        }