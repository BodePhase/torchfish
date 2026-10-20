import pandas as pd
import torch
from torch.utils.data import Dataset
import chess
import json
import os

class KasparovPolicyDataset(Dataset):
    def __init__(self, csv_file, dict_file="models/uci_dictionary.json"):
        # Carica le partite di Kasparov
        self.data = pd.read_csv(csv_file)
        
        # Mappatura standard dei pezzi (identica alla Value Network)
        self.piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Carica il vocabolario JSON generato da vocabulary.py
        if not os.path.exists(dict_file):
            raise FileNotFoundError(f"Dizionario non trovato in {dict_file}. Esegui prima vocabulary.py")
            
        with open(dict_file, "r") as f:
            vocabs = json.load(f)
            self.move_to_int = vocabs["move_to_int"]

    def __len__(self):
        return len(self.data)

    def board_to_tensor(self, board):
        # Tensore [14, 8, 8] - Identico all'altra rete per mantenere coerenza
        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = self.piece_to_channel[piece.piece_type]
                if not piece.color: 
                    channel += 6
                
                row = chess.square_rank(square)
                col = chess.square_file(square)
                tensor[channel, row, col] = 1.0

        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0

        return tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row["fen"]
        move_uci = row["move_uci"] # Es: "e2e4"
        
        board = chess.Board(fen)
        tensor_board = self.board_to_tensor(board)
        
        # Converte la mossa testuale nel suo identificativo numerico (Target)
        target_class = self.move_to_int[move_uci]
        
        # Il target è un intero (torch.long), fondamentale per la CrossEntropyLoss
        return tensor_board, torch.tensor(target_class, dtype=torch.long)