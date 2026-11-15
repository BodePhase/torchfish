import json
import chess
import pandas as pd
import torch
from torch.utils.data import Dataset

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converte chess.Board in un tensore float32 (14, 8, 8) con Traversa 1 = riga 0:
    - Canali 0..5: Pezzi Bianchi (P, N, B, R, Q, K)
    - Canali 6..11: Pezzi Neri (p, n, b, r, q, k)
    - Canale 12: Turno di gioco (1.0 se muove il Bianco, 0.0 se muove il Nero)
    - Canale 13: Diritti d'arrocco sui 4 angoli (a1, h1, a8, h8) + casa target en-passant
    """
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        ch = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        tensor[ch, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Canale 13: Diritti di arrocco (angoli Torri) + casa En-Passant (traversa 3 o 6)
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, 0, 0] = 1.0  # a1
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, 0, 7] = 1.0  # h1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, 7, 0] = 1.0  # a8
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[13, 7, 7] = 1.0  # h8

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[13, ep_rank, ep_file] = 1.0

    return tensor


class KasparovPolicyDataset(Dataset):
    def __init__(self, csv_file: str, dict_file: str):
        self.data = pd.read_csv(csv_file)
        with open(dict_file, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
        self.move_to_int = raw_dict.get("move_to_int", raw_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        board = chess.Board(row["fen"])
        board_tensor = board_to_tensor(board)
        move_idx = self.move_to_int[row["move_uci"]]
        return board_tensor, torch.tensor(move_idx, dtype=torch.long)