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

# Valori base per ordinamento mosse MVV-LVA
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Valori PeSTO separati tra Mediogioco (MG) e Finale (EG)
MG_VALUE = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}

EG_VALUE = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}

# Incremento di fase per il calcolo del Tapered Eval (Max Phase = 24)
PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
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
        "c_puct": 1.35,
        "style_bonus_cp": 22.0,
        "aggression": 1.05,
    },
}

# =============================================================================
# TABELLE POSIZIONALI PeSTO (Middlegame & Endgame, Riga 0 = Traversa 8)
# =============================================================================
_RAW_MG_PST = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    chess.BISHOP: [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    chess.QUEEN: [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
}

_RAW_EG_PST = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    chess.ROOK: [
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  -3,   3,   8,   3,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ],
    chess.QUEEN: [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    chess.KING: [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
}

# Pre-compilazione tabelle PeSTO (Valore Pezzo + Bonus Casella già sommati)
MG_WHITE = {pt: [0] * 64 for pt in _RAW_MG_PST}
MG_BLACK = {pt: [0] * 64 for pt in _RAW_MG_PST}
EG_WHITE = {pt: [0] * 64 for pt in _RAW_EG_PST}
EG_BLACK = {pt: [0] * 64 for pt in _RAW_EG_PST}

for _pt in _RAW_MG_PST:
    _mg_v = MG_VALUE[_pt]
    _eg_v = EG_VALUE[_pt]
    for _sq in range(64):
        MG_WHITE[_pt][_sq] = _mg_v + _RAW_MG_PST[_pt][_sq ^ 56]
        MG_BLACK[_pt][_sq] = _mg_v + _RAW_MG_PST[_pt][_sq]
        EG_WHITE[_pt][_sq] = _eg_v + _RAW_EG_PST[_pt][_sq ^ 56]
        EG_BLACK[_pt][_sq] = _eg_v + _RAW_EG_PST[_pt][_sq]

# Pre-compilazione Maschere Bitboard per Pedoni Passati, Colonne e Mop-Up
FILE_MASKS = [0] * 8
for _f in range(8):
    mask = 0
    for _r in range(8):
        mask |= 1 << (_r * 8 + _f)
    FILE_MASKS[_f] = mask

PASSED_MASK_WHITE = [0] * 64
PASSED_MASK_BLACK = [0] * 64
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _adj_files = FILE_MASKS[_f]
    if _f > 0:
        _adj_files |= FILE_MASKS[_f - 1]
    if _f < 7:
        _adj_files |= FILE_MASKS[_f + 1]

    _w_front = 0
    for _nr in range(_r + 1, 8):
        _w_front |= 0xFF << (_nr * 8)
    PASSED_MASK_WHITE[_sq] = _adj_files & _w_front

    _b_front = 0
    for _nr in range(0, _r):
        _b_front |= 0xFF << (_nr * 8)
    PASSED_MASK_BLACK[_sq] = _adj_files & _b_front

PASSED_BONUS_MG = [0, 5, 10, 20, 35, 65, 110, 0]
PASSED_BONUS_EG = [0, 10, 22, 45, 80, 140, 215, 0]

# Distanza dal centro per il Mop-Up (spinge il Re perdente negli angoli nei finali)
CENTER_DIST = [0] * 64
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _df = max(3 - _f, _f - 4)
    _dr = max(3 - _r, _r - 4)
    CENTER_DIST[_sq] = _df + _dr

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class SearchTimeout(Exception):
    """Interrompe in modo pulito l'Iterative Deepening allo scadere del budget di tempo."""
    pass


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
        self.tt = {}
        self.killers = {}
        self.nodes_searched = 0
        self.abort_time = 0.0

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
        print(f"[TorchFish] Motore PeSTO Tapered + LMR/NMP + KasparovNet attivo su {self.device}.")

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
    # VALUTAZIONE PeSTO TAPERED + BITBOARD PASSED PAWNS + MOP-UP ENDGAME
    # =========================================================================
    def _static_eval_cp(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -99000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        mg_score = 0
        eg_score = 0
        game_phase = 0

        w_pawns_bb = int(board.pieces(chess.PAWN, chess.WHITE))
        b_pawns_bb = int(board.pieces(chess.PAWN, chess.BLACK))

        for pt in NON_KING_PIECES:
            w_squares = board.pieces(pt, chess.WHITE)
            b_squares = board.pieces(pt, chess.BLACK)

            w_count = len(w_squares)
            b_count = len(b_squares)
            game_phase += PHASE_WEIGHT[pt] * (w_count + b_count)

            mg_w = MG_WHITE[pt]
            eg_w = EG_WHITE[pt]
            for sq in w_squares:
                mg_score += mg_w[sq]
                eg_score += eg_w[sq]

            mg_b = MG_BLACK[pt]
            eg_b = EG_BLACK[pt]
            for sq in b_squares:
                mg_score -= mg_b[sq]
                eg_score -= eg_b[sq]

            # Coppia degli Alfieri
            if pt == chess.BISHOP:
                if w_count >= 2:
                    mg_score += 30
                    eg_score += 45
                if b_count >= 2:
                    mg_score -= 30
                    eg_score -= 45

            # Pedoni Passati (controllo bitboard O(1))
            elif pt == chess.PAWN:
                for sq in w_squares:
                    if (b_pawns_bb & PASSED_MASK_WHITE[sq]) == 0:
                        rank = sq >> 3
                        mg_score += PASSED_BONUS_MG[rank]
                        eg_score += PASSED_BONUS_EG[rank]
                for sq in b_squares:
                    if (w_pawns_bb & PASSED_MASK_BLACK[sq]) == 0:
                        rank = 7 - (sq >> 3)
                        mg_score -= PASSED_BONUS_MG[rank]
                        eg_score -= PASSED_BONUS_EG[rank]

            # Torri su colonne aperte o semi-aperte
            elif pt == chess.ROOK:
                for sq in w_squares:
                    f_mask = FILE_MASKS[sq & 7]
                    if (w_pawns_bb & f_mask) == 0:
                        mg_score += 22 if (b_pawns_bb & f_mask) == 0 else 10
                for sq in b_squares:
                    f_mask = FILE_MASKS[sq & 7]
                    if (b_pawns_bb & f_mask) == 0:
                        mg_score -= 22 if (w_pawns_bb & f_mask) == 0 else 10

        # Valutazione Re (MG vs EG)
        w_king = board.king(chess.WHITE)
        b_king = board.king(chess.BLACK)
        if w_king is not None:
            mg_score += MG_WHITE[chess.KING][w_king]
            eg_score += EG_WHITE[chess.KING][w_king]
        if b_king is not None:
            mg_score -= MG_BLACK[chess.KING][b_king]
            eg_score -= EG_BLACK[chess.KING][b_king]

        # Penalità nel mediogioco per perdita diritti d'arrocco col Re ancora al centro
        if game_phase >= 14:
            if w_king in (chess.E1, chess.D1, chess.F1) and not board.has_castling_rights(chess.WHITE):
                mg_score -= 70
            if b_king in (chess.E8, chess.D8, chess.F8) and not board.has_castling_rights(chess.BLACK):
                mg_score += 70

        # Interpolazione Tapered (MG <-> EG)
        mg_phase = min(24, game_phase)
        eg_phase = 24 - mg_phase
        score = (mg_score * mg_phase + eg_score * eg_phase) // 24

        # Mop-Up Evaluation nei finali vinti (forza il Re perdente verso il bordo e avvicina il Re vincente)
        if game_phase <= 6 and w_king is not None and b_king is not None:
            kings_dist = chess.square_distance(w_king, b_king)
            if score > 300:
                score += CENTER_DIST[b_king] * 14 + (7 - kings_dist) * 8
            elif score < -300:
                score -= CENTER_DIST[w_king] * 14 + (7 - kings_dist) * 8

        return score if board.turn == chess.WHITE else -score

    # =========================================================================
    # QUIESCENCE SEARCH ESATTA
    # =========================================================================
    def _quiescence_cp(self, board: chess.Board, alpha: int, beta: int, q_depth: int = 0) -> int:
        self.nodes_searched += 1
        if (self.nodes_searched & 1023) == 0 and self.abort_time > 0 and time.perf_counter() > self.abort_time:
            raise SearchTimeout()

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

            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        else:
            if q_depth >= 7:
                return self._static_eval_cp(board)
            moves = list(board.legal_moves)

        moves.sort(
            key=lambda m: (
                PIECE_VALUES.get(board.piece_type_at(m.to_square) or chess.PAWN, 100) * 10
                - PIECE_VALUES.get(board.piece_type_at(m.from_square) or chess.PAWN, 100)
                + (900 if m.promotion else 0)
            ),
            reverse=True
        )

        for move in moves:
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
    # NEGAMAX ALPHA-BETA CON NMP, LMR, KILLER MOVES & CHECK EXTENSION
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
                if board.is_attacked_by(enemy, m.from_square):
                    s += 600
                if board.is_attacked_by(enemy, m.to_square):
                    s -= 400
            return s

        return sorted(moves, key=score_move, reverse=True)

    def _has_non_pawn_material(self, board: chess.Board, color: bool) -> bool:
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if board.pieces(pt, color):
                return True
        return False

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True
    ) -> int:
        self.nodes_searched += 1
        if (self.nodes_searched & 1023) == 0 and self.abort_time > 0 and time.perf_counter() > self.abort_time:
            raise SearchTimeout()

        if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
            return 0
        if board.is_checkmate():
            return -(99000 - ply * 100)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

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

        # 1. Null Move Pruning (NMP)
        if (
            allow_null
            and not in_check
            and ply > 0
            and depth >= 3
            and beta < 90000
            and self._has_non_pawn_material(board, board.turn)
        ):
            eval_now = self._static_eval_cp(board)
            if eval_now >= beta:
                board.push(chess.Move.null())
                null_score = -self._negamax(
                    board, depth - 1 - 2, -beta, -beta + 1, ply + 1, allow_null=False
                )
                board.pop()
                if null_score >= beta:
                    return beta

        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves, tt_move, ply)

        best_val = -999999
        best_move = ordered_moves[0]

        for move_idx, move in enumerate(ordered_moves):
            is_capture = board.is_capture(move)
            is_promo = bool(move.promotion)
            gives_chk = board.gives_check(move)

            board.push(move)

            # 2. Late Move Reductions (LMR) + Principal Variation Search (PVS)
            if move_idx == 0:
                val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                do_lmr = (
                    depth >= 3
                    and move_idx >= 3
                    and not in_check
                    and not is_capture
                    and not is_promo
                    and not gives_chk
                )
                reduction = 1 if do_lmr else 0
                if do_lmr and move_idx >= 8:
                    reduction = 2

                # Ricerca con finestra ridotta (Null Window) ed eventuale riduzione LMR
                val = -self._negamax(
                    board, max(0, depth - 1 - reduction), -alpha - 1, -alpha, ply + 1
                )

                # Se la mossa ridotta supera alpha, riesegui la ricerca a profondità e finestra piene!
                if val > alpha and (reduction > 0 or val < beta):
                    val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if val > best_val:
                best_val = val
                best_move = move
            if val > alpha:
                alpha = val
            if alpha >= beta:
                if not is_capture:
                    k1, k2 = self.killers.get(ply, (None, None))
                    if move != k1:
                        self.killers[ply] = (move, k1)
                break

        tt_flag = TT_EXACT
        if best_val <= alpha_orig:
            tt_flag = TT_UPPER
        elif best_val >= beta:
            tt_flag = TT_LOWER

        if len(self.tt) > 200000:
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

        self.abort_time = 0.0
        cp_mover = self._quiescence_cp(board, -99999, 99999, 0)
        cp_white = cp_mover if board.turn == chess.WHITE else -cp_mover
        return {
            "eval_white": round(cp_white / 100.0, 2),
            "mate_in": None
        }

    # =========================================================================
    # VALUTAZIONE PRINCIPALE: ITERATIVE DEEPENING (DEPTH 1..4) + KASPAROV POLICY
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
        self.nodes_searched = 0

        # 3. Calcola i bonus posizionali/stile statici per ogni mossa legale alla radice
        root_adjustments = {}
        for move in legal_moves:
            uci = move.uci()
            prior = policy_priors.get(move, 1e-5)
            moving_piece = board.piece_at(move.from_square)
            is_castling_move = board.is_castling(move)
            is_cap = board.is_capture(move)

            board.push(move)
            next_fen_key = " ".join(board.fen().split()[:4])

            adj = 0.0
            if had_castling_rights and not is_castling_move and not board.has_castling_rights(ai_color):
                adj -= 140.0
            if moving_piece and moving_piece.piece_type == chess.ROOK and not is_castling_move and board.fullmove_number <= 12:
                if not is_cap and not board.is_check():
                    adj -= 95.0

            if is_castling_move:
                adj += 55.0

            if board.fullmove_number <= 10 and moving_piece:
                if moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    from_rank = chess.square_rank(move.from_square)
                    to_file = chess.square_file(move.to_square)
                    if (ai_is_white and from_rank == 0) or (not ai_is_white and from_rank == 7):
                        adj += 22.0
                    if moving_piece.piece_type == chess.KNIGHT and to_file in (0, 7) and not is_cap:
                        adj -= 40.0
                elif moving_piece.piece_type == chess.PAWN:
                    if chess.square_file(move.from_square) in (2, 3, 4):
                        adj += 16.0

            if next_fen_key in self.recent_positions:
                adj -= 120.0

            board.pop()

            style_cp = style_max_cp * prior
            book_cp = (75.0 * (book_hits[uci] / total_book)) if (uci in book_hits and total_book > 0) else 0.0
            root_adjustments[move] = (adj, style_cp, book_cp)

        # 4. Iterative Deepening (fino a Depth 4 = 5 ply pieni + Quiescence) con LMR + NMP
        start_time = time.perf_counter()
        time_budget = 0.60  # 600ms max per mossa
        self.abort_time = start_time + time_budget

        tactical_scores = {m: 0 for m in legal_moves}
        ordered_root_moves = self._order_moves(board, legal_moves, None, 0)
        reached_depth = 1

        for current_depth in range(1, 5):
            # A depth=1 disabilita il timeout per garantire almeno 2 ply completi su tutte le mosse
            self.abort_time = 0.0 if current_depth == 1 else (start_time + time_budget)

            best_tactical_so_far = -999999
            new_tactical_scores = {}
            completed_iteration = True

            try:
                for idx, move in enumerate(ordered_root_moves):
                    board.push(move)

                    if idx < top_k or best_tactical_so_far < -50000:
                        opp_beta = 999999
                    else:
                        opp_beta = -(best_tactical_so_far - 200)

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
            except SearchTimeout:
                board = chess.Board(fen)
                completed_iteration = False
                # Se abbiamo già valutato le mosse principali prima del timeout, aggiorna quelle completate
                if len(new_tactical_scores) >= top_k:
                    for m, v in new_tactical_scores.items():
                        tactical_scores[m] = v
                break

            if completed_iteration:
                tactical_scores = new_tactical_scores
                reached_depth = current_depth + 1
                ordered_root_moves.sort(
                    key=lambda m: tactical_scores[m] + sum(root_adjustments[m]),
                    reverse=True
                )
                # Se abbiamo trovato uno scacco matto forzato, non serve cercare oltre
                if best_tactical_so_far > 90000:
                    break

        self.abort_time = 0.0

        # 5. Combina il punteggio tattico con lo stile della Leggenda (Blunder Lock attivo)
        best_pure_tactical = max(tactical_scores.values())
        scored_moves = []

        for move in legal_moves:
            san = board.san(move)
            uci = move.uci()
            prior = policy_priors.get(move, 1e-5)
            t_cp = tactical_scores[move]
            adj, style_cp, book_cp = root_adjustments[move]

            if best_pure_tactical - t_cp >= 85:
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
            ply_to_mate = max(1, (99000 - abs(top_slice[0]["tactical_cp"])) // 100)
            moves_to_mate = math.ceil(ply_to_mate / 2)
            mate_in = moves_to_mate if (top_slice[0]["tactical_cp"] > 0) == ai_is_white else -moves_to_mate

        return {
            "best_move_san": best["san"],
            "best_move_uci": best["uci"],
            "confidence": best["prob"],
            "top_moves": top_moves,
            "is_book_move": top_slice[0]["in_book"],
            "legend": legend_id,
            "eval_white": best["eval_white"],
            "mate_in": mate_in,
            "search_depth": reached_depth,
            "nodes": self.nodes_searched
        }