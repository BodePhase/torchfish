import os
import csv
import json
import math
import time
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
import chess

from models.policy_net import KasparovPolicyNet

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

NON_KING_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

# Profili comportamentali delle Leggende
LEGENDS_PROFILES = {
    "kasparov": {
        "name": "Garry Kasparov",
        "title": "Il Mostro di Baku",
        "style_desc": "Aggressivo, dinamico, iniziativa costante",
        "weights_path": "weights/kasparov_policy.pth",
        "book_path": "data/gm_kasparov/kasparov_moves.csv",
        "style_bonus_cp": 32.0,
        "aggression": 1.20,
    },
    "tal": {
        "name": "Mikhail Tal",
        "title": "Il Mago di Riga",
        "style_desc": "Sacrifici audaci, attacco diretto al Re",
        "weights_path": "weights/tal_policy.pth",
        "book_path": "data/gm_tal/tal_moves.csv",
        "style_bonus_cp": 38.0,
        "aggression": 1.40,
    },
    "karpov": {
        "name": "Anatoly Karpov",
        "title": "Il Boa Costrittore",
        "style_desc": "Profilassi, solidità assoluta, zero rischi",
        "weights_path": "weights/karpov_policy.pth",
        "book_path": "data/gm_karpov/karpov_moves.csv",
        "style_bonus_cp": 25.0,
        "aggression": 0.90,
    },
    "carlsen": {
        "name": "Magnus Carlsen",
        "title": "Il Mozart degli Scacchi",
        "style_desc": "Precisione universale, pressione nei finali",
        "weights_path": "weights/carlsen_policy.pth",
        "book_path": "data/gm_carlsen/carlsen_moves.csv",
        "style_bonus_cp": 22.0,
        "aggression": 1.05,
    },
}

# Tabelle posizionali (riga 0 = Traversa 8, riga 7 = Traversa 1)
_RAW_PST = {
    chess.PAWN: [
         0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
         5,   5,  10,  25,  25,  10,   5,   5,
         0,   0,   5,  22,  22,   5,   0,   0,
         5,  -5, -10,   0,   0, -10,  -5,   5,
         5,  10,  10, -20, -20,  10,  10,   5,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  12,  18,  18,  12,   0, -30,
        -30,   5,  18,  22,  22,  18,   5, -30,
        -30,   0,  18,  22,  22,  18,   0, -30,
        -30,   5,  12,  15,  15,  12,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -35, -30, -30, -30, -30, -35, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   8,  10,  14,  14,  10,   8, -10,
        -10,   4,  14,  14,  14,  14,   4, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,  12,   0,   0,   0,   0,  12, -10,
        -20, -10, -12, -10, -10, -12, -10, -20,
    ],
    chess.ROOK: [
          0,   0,   0,   0,   0,   0,   0,   0,
         10,  15,  15,  15,  15,  15,  15,  10,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          0,  -5,   0,   8,   8,   5, -10,   0
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   8,   8,   5,   0,  -5,
          0,   0,   5,   8,   8,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         15,  15,  -5, -10, -10,  -5,  15,  15,
         20,  35,  15, -15,   0, -10,  35,  20
    ]
}

# Pre-calcola le tabelle PST indicizzate direttamente da sq (0..63) per Bianco e Nero
PST_WHITE = {pt: [0] * 64 for pt in _RAW_PST}
PST_BLACK = {pt: [0] * 64 for pt in _RAW_PST}
for _pt, _table in _RAW_PST.items():
    for _sq in range(64):
        PST_WHITE[_pt][_sq] = _table[_sq ^ 56]
        PST_BLACK[_pt][_sq] = _table[_sq]

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class KasparovEngine:
    def __init__(
        self,
        model_path: str = "weights/kasparov_policy.pth",
        value_model_path: str = "best_chess_model.pth",
        dict_path: str = "models/uci_dictionary.json",
        book_csv_path: str = "data/gm_kasparov/kasparov_moves.csv"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_model_path = model_path
        self.recent_positions = deque(maxlen=16)
        self.tt = {}  # Transposition Table
        self.killers = {}  # Killer heuristic per ordinamento mosse

        # 1. Caricamento Robusto del Dizionario UCI
        with open(dict_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)

        if isinstance(raw_dict, dict):
            if "move_to_int" in raw_dict or "move_to_idx" in raw_dict:
                sub_dict = raw_dict.get("move_to_int", raw_dict.get("move_to_idx"))
                self.move_to_idx = {str(k): int(v) for k, v in sub_dict.items()}
                self.idx_to_move = {int(v): str(k) for k, v in sub_dict.items()}
            else:
                first_key = list(raw_dict.keys())[0]
                if str(first_key).isdigit():
                    self.idx_to_move = {int(k): str(v) for k, v in raw_dict.items()}
                    self.move_to_idx = {str(v): int(k) for k, v in raw_dict.items()}
                else:
                    self.move_to_idx = {str(k): int(v) for k, v in raw_dict.items()}
                    self.idx_to_move = {int(v): str(k) for k, v in raw_dict.items()}
        else:
            self.idx_to_move = {i: str(m) for i, m in enumerate(raw_dict)}
            self.move_to_idx = {str(m): i for i, m in enumerate(raw_dict)}

        self.num_classes = len(self.move_to_idx)

        # 2. Usa l'encoder originale di dataset_policy.py se presente
        self.custom_fen_encoder = None
        try:
            import dataset_policy
            for fn_name in ("fen_to_tensor", "board_to_tensor", "encode_board", "encode_fen"):
                if hasattr(dataset_policy, fn_name):
                    self.custom_fen_encoder = getattr(dataset_policy, fn_name)
                    break
        except Exception:
            pass

        # 3. Carica i modelli Policy per le Leggende
        self.policy_models = {}
        self.opening_books = {}
        self.policy_channels = 12

        base_net = self._load_policy_model(model_path)
        self.policy_models["kasparov"] = base_net

        for legend_id, cfg in LEGENDS_PROFILES.items():
            w_path = cfg["weights_path"]
            if legend_id != "kasparov" and os.path.exists(w_path):
                try:
                    self.policy_models[legend_id] = self._load_policy_model(w_path)
                except Exception:
                    self.policy_models[legend_id] = base_net
            else:
                self.policy_models[legend_id] = base_net

            b_path = cfg["book_path"] if os.path.exists(cfg["book_path"]) else book_csv_path
            self.opening_books[legend_id] = self._load_book(b_path) if os.path.exists(b_path) else {}

        self.value_net = None
        print(f"[TorchFish] Motore Tattico Iterative-Deepening + KasparovNet pronto su {self.device}.")

    def _load_policy_model(self, path: str):
        try:
            net = KasparovPolicyNet(num_classes=self.num_classes).to(self.device)
        except TypeError:
            net = KasparovPolicyNet().to(self.device)

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
        else:
            net.load_state_dict(ckpt)
        net.eval()

        first_param = next(net.parameters())
        self.policy_channels = first_param.shape[1] if first_param.ndim == 4 else 12
        return net

    def _load_book(self, csv_path: str) -> dict:
        book = {}
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fen_key = " ".join(row["fen"].split()[:4])
                    uci = row["move_uci"].strip()
                    if fen_key not in book:
                        book[fen_key] = {}
                    book[fen_key][uci] = book[fen_key].get(uci, 0) + 1
        except Exception:
            pass
        return book

    def get_legends_info(self):
        return [
            {
                "id": lid,
                "name": cfg["name"],
                "title": cfg["title"],
                "style_desc": cfg["style_desc"],
                "has_custom_weights": os.path.exists(cfg["weights_path"])
            }
            for lid, cfg in LEGENDS_PROFILES.items()
        ]

    def _board_to_tensor(self, board: chess.Board, channels: int) -> torch.Tensor:
        if self.custom_fen_encoder is not None:
            try:
                t = self.custom_fen_encoder(board.fen())
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t).float()
                if t.shape[0] == channels:
                    return t
            except Exception:
                pass

        tensor = np.zeros((channels, 8, 8), dtype=np.float32)
        piece_order = [
            (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
            (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
            (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
            (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
        ]
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                ch = piece_order.index((piece.piece_type, piece.color))
                if ch < channels:
                    tensor[ch, 7 - (sq // 8), sq % 8] = 1.0
        if channels >= 13:
            tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        return torch.from_numpy(tensor)

    # =========================================================================
    # VALUTAZIONE STATICA BITBOARD ULTRA-VELOCE (Pura, senza distorsioni d'orizzonte)
    # =========================================================================
    def _static_eval_cp(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -99000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        non_pawn_mat = 0

        for pt in NON_KING_PIECES:
            val = PIECE_VALUES[pt]
            w_squares = board.pieces(pt, chess.WHITE)
            b_squares = board.pieces(pt, chess.BLACK)

            w_count = len(w_squares)
            b_count = len(b_squares)
            score += val * (w_count - b_count)

            if pt != chess.PAWN:
                non_pawn_mat += val * (w_count + b_count)
                if pt == chess.BISHOP:
                    if w_count >= 2:
                        score += 30
                    if b_count >= 2:
                        score -= 30

            pst_w = PST_WHITE[pt]
            pst_b = PST_BLACK[pt]
            for sq in w_squares:
                score += pst_w[sq]
            for sq in b_squares:
                score -= pst_b[sq]

        # Valutazione posizione dei Re
        w_king = board.king(chess.WHITE)
        b_king = board.king(chess.BLACK)
        if w_king is not None:
            score += PST_WHITE[chess.KING][w_king]
        if b_king is not None:
            score -= PST_BLACK[chess.KING][b_king]

        # In apertura/mediogioco penalizza la perdita dei diritti d'arrocco col Re ancora al centro
        if non_pawn_mat >= 2600:
            if w_king in (chess.E1, chess.D1, chess.F1) and not board.has_castling_rights(chess.WHITE):
                score -= 75
            if b_king in (chess.E8, chess.D8, chess.F8) and not board.has_castling_rights(chess.BLACK):
                score += 75

        return score if board.turn == chess.WHITE else -score

    # =========================================================================
    # QUIESCENCE SEARCH ESATTA (Risolve tutte le catene di catture e promozioni)
    # =========================================================================
    def _quiescence_cp(self, board: chess.Board, alpha: int, beta: int, q_depth: int = 0) -> int:
        if board.is_checkmate():
            return -(99000 - q_depth * 100)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        in_check = board.is_check()

        if not in_check:
            stand_pat = self._static_eval_cp(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if q_depth >= 6:
                return stand_pat

            # Genera catture e promozioni
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        else:
            # Sotto scacco bisogna generare tutte le evasioni legali
            if q_depth >= 7:
                return self._static_eval_cp(board)
            moves = list(board.legal_moves)

        # Ordina secondo MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        moves.sort(
            key=lambda m: (
                PIECE_VALUES.get(board.piece_type_at(m.to_square) or chess.PAWN, 100) * 10
                - PIECE_VALUES.get(board.piece_type_at(m.from_square) or chess.PAWN, 100)
                + (900 if m.promotion else 0)
            ),
            reverse=True
        )

        for move in moves:
            # Delta pruning: salta catture che non possono in alcun modo migliorare alpha
            if not in_check and not move.promotion and q_depth >= 2:
                victim_pt = board.piece_type_at(move.to_square)
                victim_val = PIECE_VALUES.get(victim_pt, 100) if victim_pt else 100
                if stand_pat + victim_val + 200 < alpha:
                    continue

            board.push(move)
            score = -self._quiescence_cp(board, -beta, -alpha, q_depth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # =========================================================================
    # NEGAMAX ALPHA-BETA CON TRANSPOSITION TABLE, KILLER MOVES & CHECK EXTENSION
    # =========================================================================
    def _order_moves(self, board: chess.Board, moves: list, tt_move: chess.Move, ply: int) -> list:
        killer_pair = self.killers.get(ply, (None, None))
        enemy = not board.turn

        def score_move(m: chess.Move) -> int:
            if m == tt_move:
                return 100000
            if board.is_capture(m):
                victim = PIECE_VALUES.get(board.piece_type_at(m.to_square) or chess.PAWN, 100)
                attacker = PIECE_VALUES.get(board.piece_type_at(m.from_square) or chess.PAWN, 100)
                return 10000 + victim * 10 - attacker
            if m.promotion:
                return 9500
            if m == killer_pair[0]:
                return 8000
            if m == killer_pair[1]:
                return 7500

            s = 0
            moving_pt = board.piece_type_at(m.from_square)
            if moving_pt and moving_pt != chess.KING:
                # Priorità alta al salvataggio di pezzi attualmente sotto attacco!
                if board.is_attacked_by(enemy, m.from_square):
                    s += 600
                # Penalizza nell'ordinamento lo spostamento su case già attaccate
                if board.is_attacked_by(enemy, m.to_square):
                    s -= 400
            return s

        return sorted(moves, key=score_move, reverse=True)

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
            return 0
        if board.is_checkmate():
            return -(99000 - ply * 100)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Check Extension: non entrare in Quiescenza mentre si è sotto scacco!
        in_check = board.is_check()
        if in_check and ply < 8:
            depth += 1

        if depth <= 0:
            return self._quiescence_cp(board, alpha, beta, 0)

        alpha_orig = alpha
        pos_key = getattr(board, "_transposition_key", board.fen)()
        tt_entry = self.tt.get(pos_key)
        tt_move = None

        if tt_entry is not None:
            tt_depth, tt_flag, tt_score, tt_move = tt_entry
            if tt_depth >= depth and ply > 0:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWER and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == TT_UPPER and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return tt_score

        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves, tt_move, ply)

        best_val = -999999
        best_move = ordered_moves[0]

        for move in ordered_moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if val > best_val:
                best_val = val
                best_move = move
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # Salva Killer Move se non è una cattura
                if not board.is_capture(move):
                    k1, k2 = self.killers.get(ply, (None, None))
                    if move != k1:
                        self.killers[ply] = (move, k1)
                break

        tt_flag = TT_EXACT
        if best_val <= alpha_orig:
            tt_flag = TT_UPPER
        elif best_val >= beta:
            tt_flag = TT_LOWER

        if len(self.tt) > 150000:
            self.tt.clear()
        self.tt[pos_key] = (depth, tt_flag, best_val, best_move)

        return best_val

    def _get_policy_priors(self, board: chess.Board, legend_id: str) -> dict:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        net = self.policy_models.get(legend_id, self.policy_models["kasparov"])
        with torch.no_grad():
            inp = self._board_to_tensor(board, self.policy_channels).unsqueeze(0).to(self.device)
            logits = net(inp).squeeze(0)
            probs = F.softmax(logits, dim=-1)

        priors = {}
        total = 0.0
        for move in legal_moves:
            uci = move.uci()
            p = float(probs[self.move_to_idx[uci]].item()) if uci in self.move_to_idx else 1e-5
            priors[move] = p
            total += p

        norm = total if total > 0 else 1.0
        for move in priors:
            priors[move] /= norm
        return priors

    def evaluate_only(self, fen: str, legend: str = "kasparov") -> dict:
        board = chess.Board(fen)
        if board.is_checkmate():
            winner_sign = -1 if board.turn == chess.WHITE else 1
            return {"eval_white": float(winner_sign * 99.0), "mate_in": winner_sign * 1}
        if board.is_game_over():
            return {"eval_white": 0.0, "mate_in": None}

        cp_mover = self._quiescence_cp(board, -99999, 99999, 0)
        cp_white = cp_mover if board.turn == chess.WHITE else -cp_mover
        return {
            "eval_white": round(cp_white / 100.0, 2),
            "mate_in": None
        }

    # =========================================================================
    # VALUTAZIONE PRINCIPALE: ITERATIVE DEEPENING + KASPAROV POLICY
    # =========================================================================
    def evaluate_position(
        self,
        fen: str,
        top_k: int = 3,
        num_simulations: int = 64,
        legend: str = "kasparov"
    ):
        legend_id = legend.lower() if legend.lower() in LEGENDS_PROFILES else "kasparov"
        profile = LEGENDS_PROFILES[legend_id]

        board = chess.Board(fen)
        if board.is_game_over():
            eval_info = self.evaluate_only(fen, legend_id)
            return {
                "best_move_san": None,
                "best_move_uci": None,
                "confidence": 0.0,
                "top_moves": [],
                "is_book_move": False,
                "legend": legend_id,
                "eval_white": eval_info["eval_white"],
                "mate_in": eval_info["mate_in"],
            }

        if board.fullmove_number <= 2:
            self.recent_positions.clear()
            self.tt.clear()
        current_fen_key = " ".join(fen.split()[:4])
        self.recent_positions.append(current_fen_key)

        legal_moves = list(board.legal_moves)
        ai_color = board.turn
        ai_is_white = ai_color == chess.WHITE
        had_castling_rights = board.has_castling_rights(ai_color)

        # 1. Check istantaneo Matto in 1
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                san = board.san(move)
                return {
                    "best_move_san": san,
                    "best_move_uci": move.uci(),
                    "confidence": 100.0,
                    "top_moves": [{"san": san, "uci": move.uci(), "prob": 100.0}],
                    "is_book_move": False,
                    "legend": legend_id,
                    "eval_white": 99.0 if ai_is_white else -99.0,
                    "mate_in": 1 if ai_is_white else -1,
                }
            board.pop()

        # 2. Prior dalla Policy Network e dall'Opening Book storico
        policy_priors = self._get_policy_priors(board, legend_id)
        book = self.opening_books.get(legend_id, {})
        book_hits = book.get(current_fen_key, {})
        total_book = sum(book_hits.values()) if book_hits else 0

        style_max_cp = profile["style_bonus_cp"]
        self.killers.clear()

        # 3. Calcola i bonus posizionali/stile statici per ogni mossa legale alla radice
        root_ adjustments = {}
        for move in legal_moves:
            uci = move.uci()
            prior = policy_priors.get(move, 1e-5)
            moving_piece = board.piece_at(move.from_square)
            is_castling_move = board.is_castling(move)

            board.push(move)
            next_fen_key = " ".join(board.fen().split()[:4])

            adj = 0.0
            # Vieta di bruciare i diritti d'arrocco muovendo Re o Torre senza motivo (es. Rg8)
            if had_castling_rights and not is_castling_move and not board.has_castling_rights(ai_color):
                adj -= 140.0
            if moving_piece and moving_piece.piece_type == chess.ROOK and not is_castling_move and board.fullmove_number <= 12:
                if not board.is_capture(move) and not board.is_check():
                    adj -= 100.0

            if is_castling_move:
                adj += 55.0

            # Sviluppo in apertura
            if board.fullmove_number <= 10 and moving_piece:
                if moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    from_rank = chess.square_rank(move.from_square)
                    to_file = chess.square_file(move.to_square)
                    if (ai_is_white and from_rank == 0) or (not ai_is_white and from_rank == 7):
                        adj += 25.0
                    if moving_piece.piece_type == chess.KNIGHT and to_file in (0, 7) and not board.is_capture(move):
                        adj -= 45.0
                elif moving_piece.piece_type == chess.PAWN:
                    if chess.square_file(move.from_square) in (2, 3, 4):
                        adj += 18.0

            # Penalizza ripetizioni inutili della posizione
            if next_fen_key in self.recent_positions:
                adj -= 120.0

            board.pop()

            # Bonus stile Kasparov (max 32-38 cp: decide tra mosse sane, non può mai superare la perdita di un pedone = 100 cp)
            style_cp = style_max_cp * prior
            book_cp = (75.0 * (book_hits[uci] / total_book)) if (uci in book_hits and total_book > 0) else 0.0

            root_adjustments[move] = (adj, style_cp, book_cp)

        # 4. Iterative Deepening (Depth 2 -> Depth 3) su TUTTE le mosse legali (nessuna profondità mista!)
        start_time = time.perf_counter()
        time_budget = 0.55  # 550ms max per garantire sia profondità 3 piena sia reattività UI
        tactical_scores = {m: 0 for m in legal_moves}

        ordered_root_moves = self._order_moves(board, legal_moves, None, 0)

        for current_depth in (1, 2):  # depth=2 dopo board.push(move) = 3 ply pieni + Quiescence (fino a 9 ply tattici!)
            if current_depth > 1 and (time.perf_counter() - start_time) > time_budget:
                break

            best_tactical_so_far = -999999
            new_tactical_scores = {}

            for idx, move in enumerate(ordered_root_moves):
                board.push(move)

                # Finestra Alpha-Beta alla radice: taglia all'istante i blunder che perdono > 220 cp rispetto alla migliore mossa già trovata
                if idx < top_k or best_tactical_so_far < -50000:
                    opp_beta = 999999
                else:
                    opp_beta = -(best_tactical_so_far - 220)

                val = -self._negamax(
                    board,
                    depth=current_depth,
                    alpha=-999999,
                    beta=max(-999998, opp_beta),
                    ply=1
                )
                board.pop()

                new_tactical_scores[move] = val
                if val > best_tactical_so_far:
                    best_tactical_so_far = val

            tactical_scores = new_tactical_scores
            # Riordina le mosse alla radice per l'iterazione successiva (PV first!)
            ordered_root_moves.sort(
                key=lambda m: tactical_scores[m] + sum(root_adjustments[m]),
                reverse=True
            )

        # 5. Combina il punteggio tattico reale (a parità di profondità!) con lo stile della Leggenda
        best_pure_tactical = max(tactical_scores.values())
        scored_moves = []

        for move in legal_moves:
            san = board.san(move)
            uci = move.uci()
            prior = policy_priors.get(move, 1e-5)
            t_cp = tactical_scores[move]
            adj, style_cp, book_cp = root_adjustments[move]

            # Se una mossa perde materiale rispetto alla mossa tatticamente migliore (> 90 cp = quasi un pedone),
            # azzera i bonus di stile/apertura così il motore non sceglie MAI un blunder per "stile"!
            if best_pure_tactical - t_cp >= 90:
                effective_bonus = 0.0
            else:
                effective_bonus = adj + style_cp + book_cp

            total_score = t_cp + effective_bonus
            eval_white_pawns = round((t_cp if ai_is_white else -t_cp) / 100.0, 2)

            scored_moves.append({
                "san": san,
                "uci": uci,
                "tactical_cp": t_cp,
                "eval_white": eval_white_pawns,
                "prior": prior,
                "score": total_score,
                "in_book": uci in book_hits
            })

        scored_moves.sort(key=lambda x: x["score"], reverse=True)
        top_slice = scored_moves[:top_k]

        best_move_obj = chess.Move.from_uci(top_slice[0]["uci"])
        board.push(best_move_obj)
        self.recent_positions.append(" ".join(board.fen().split()[:4]))
        board.pop()

        max_s = top_slice[0]["score"]
        exps = [math.exp(max(-15.0, (m["score"] - max_s) / 45.0)) for m in top_slice]
        sum_exps = sum(exps) or 1.0

        top_moves = [
            {
                "san": m["san"],
                "uci": m["uci"],
                "prob": round((e_val / sum_exps) * 100, 1),
                "eval_white": m["eval_white"]
            }
            for m, e_val in zip(top_slice, exps)
        ]

        best = top_moves[0]
        mate_in = None
        if abs(top_slice[0]["tactical_cp"]) > 90000:
            mate_in = 2 if (top_slice[0]["tactical_cp"] > 0) == ai_is_white else -2

        return {
            "best_move_san": best["san"],
            "best_move_uci": best["uci"],
            "confidence": best["prob"],
            "top_moves": top_moves,
            "is_book_move": top_slice[0]["in_book"],
            "legend": legend_id,
            "eval_white": best["eval_white"],
            "mate_in": mate_in
        }