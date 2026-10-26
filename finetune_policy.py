import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.policy_net import KasparovPolicyNet
from dataset_policy import KasparovPolicyDataset

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "data/gm_kasparov/kasparov_tactical_augmented.csv"
    dict_path = "models/uci_dictionary.json"
    weights_path = "weights/kasparov_policy.pth"

    dataset = KasparovPolicyDataset(csv_file=csv_path, dict_file=dict_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    # Ricava il numero di classi in modo flessibile
    try:
        num_classes = len(dataset.move_to_int)
    except AttributeError:
        num_classes = len(dataset.move_to_idx)

    # Inizializza il modello (gestendo varianti di __init__)
    try:
        model = KasparovPolicyNet(num_classes=num_classes).to(device)
    except TypeError:
        model = KasparovPolicyNet().to(device)

    # Carica i pesi originali
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Learning rate basso per non dimenticare lo stile di Kasparov
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 4
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {(correct/total)*100:.2f}%")

    # Salva i pesi aggiornati sovrascrivendo quelli vecchi
    torch.save(model.state_dict(), weights_path)
    print(f"\n[Fine-Tuning] Completato! Pesi aggiornati e salvati con successo in {weights_path}")

if __name__ == "__main__":
    finetune()