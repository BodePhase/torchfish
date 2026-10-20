import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import ChessEvaluationDataset
from model import ChessValueNetwork
import os

def train():
    # Iperparametri ottimizzati per RTX 3080 Ti su WSL2
    BATCH_SIZE = 1024 
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 8 # Sfrutta la CPU per preparare i tensori velocemente
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Inizializzazione Dataset e DataLoader
    # NB: Assicurati di avere un file 'chess_data.csv' con intestazioni FEN, Score
    dataset = ChessEvaluationDataset("chess_data.csv")
    
    # Split 90/10 per training e validazione
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = ChessValueNetwork(num_blocks=6, num_channels=256).to(device)
    
    # Smooth L1 Loss è eccellente per la regressione, è meno sensibile agli outlier rispetto al MSE
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Scaler per Mixed Precision (PyTorch 2.12+ syntax)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (boards, targets) in enumerate(train_loader):
            boards, targets = boards.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward in Mixed Precision
            with autocast('cuda'):
                outputs = model(boards)
                loss = criterion(outputs, targets)
                
            # Backward e Ottimizzazione
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for boards, targets in val_loader:
                boards, targets = boards.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(boards)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Salvataggio del checkpoint migliore
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_chess_model.pth")
            print(">>> Nuovo modello migliore salvato!")

if __name__ == "__main__":
    train()