import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import chess
import os

class ChessEvaluationDataset(Dataset):
    def __init__(self, csv_file):
        # Definiamo la mappatura dei pezzi in canali (serve SEMPRE)
        self.piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Gestiamo il bypass per l'inferenza in modo pulito
        if csv_file == "dummy.csv" or not os.path.exists(csv_file):
            self.data = pd.DataFrame() 
        else:
            self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def board_to_tensor(self, board):
        # Tensore [14, 8, 8] - 12 per i pezzi, 2 per informazioni extra (es. turno)
        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = self.piece_to_channel[piece.piece_type]
                # Se il pezzo è nero, usiamo i canali da 6 a 11
                if not piece.color: 
                    channel += 6
                
                # Calcoliamo riga (rank) e colonna (file)
                row = chess.square_rank(square)
                col = chess.square_file(square)
                tensor[channel, row, col] = 1.0

        # Canale 12: Turno (1 per il Bianco, 0 per il Nero)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        # Canale 13: Esempio per informazioni di arrocco o simili (opzionale)
        # tensor[13, :, :] = ... 

        return tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row["fen"]
        score = row["normalized_score"]

        board = chess.Board(fen)
        tensor = self.board_to_tensor(board)
        
        # Restituiamo il tensore della scacchiera e il target (punteggio)
        return tensor, torch.tensor([score], dtype=torch.float32)