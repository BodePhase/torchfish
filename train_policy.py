import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# Importiamo le classi dai file che hai appena creato
# Assicurati che i nomi dei file/cartelle corrispondano alla tua struttura
from dataset_policy import KasparovPolicyDataset
from models.policy_net import KasparovPolicyNet 

def train():
    # 1. Configurazione Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware in uso: {device}")

    # 2. Iperparametri
    # Uso una RTX 3080 Ti con 12GB di VRAM: posso quindi usare batch enormi
    # dato che ~75.000 righe di Kasparov sono molto leggere
    BATCH_SIZE = 1024 
    LEARNING_RATE = 0.001
    EPOCHS = 30
    
    # 3. Preparazione Dati
    print("Caricamento dataset in memoria...")
    dataset = KasparovPolicyDataset(
        csv_file="data/gm_kasparov/kasparov_moves.csv", 
        dict_file="models/uci_dictionary.json"
    )
    
    # num_workers=4 e pin_memory=True garantiscono che la CPU invii
    # i dati alla GPU senza creare colli di bottiglia
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 4. Inizializzazione Modello, Loss e Ottimizzatore
    model = KasparovPolicyNet(num_classes=4544).to(device)
    
    # CRITICO: La CrossEntropyLoss calcola internamente il Softmax sui logits crudi
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Inizio addestramento su {len(dataset)} posizioni di Kasparov...")
    
    # Crea la cartella per i pesi se non esiste
    os.makedirs("weights", exist_ok=True)
    
    # 5. Ciclo di Addestramento
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for batch_idx, (boards, target_moves) in enumerate(dataloader):
            boards, target_moves = boards.to(device), target_moves.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(boards) # Output: [Batch, 4544]
            
            # Calcolo Loss
            loss = criterion(outputs, target_moves)
            
            # Backward pass (Aggiornamento dei pesi)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calcolo Accuratezza (Quante volte indovina esattamente la mossa?)
            # torch.max restituisce (valori_massimi, indici_dei_valori_massimi)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target_moves.size(0)
            correct_predictions += (predicted == target_moves).sum().item()
        
        # Statistiche di fine Epoca
        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        elapsed = time.time() - start_time
        
        print(f"Epoca [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuratezza: {accuracy:.2f}% | Tempo: {elapsed:.1f}s")
    
    # 6. Salvataggio Pesi
    torch.save(model.state_dict(), "weights/kasparov_policy.pth")
    print("Addestramento completato! Pesi salvati in weights/kasparov_policy.pth")

if __name__ == "__main__":
    train()