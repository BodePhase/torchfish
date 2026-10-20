import torch
import torch.nn as nn
import torch.nn.functional as F

class KasparovPolicyNet(nn.Module):
    def __init__(self, num_classes=4544):
        super(KasparovPolicyNet, self).__init__()
        
        # 1. Blocco Estrattore di Feature (Convoluzioni)
        # Input: [Batch, 14, 8, 8] - 14 canali (pezzi + info), scacchiera 8x8
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # 2. Blocco Classificatore (Fully Connected)
        # La dimensione dopo le convoluzioni è 256 canali * 8 * 8 = 16384
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=1024)
        
        # L'output layer DEVE avere esattamente lo stesso numero di neuroni del vocabolario
        self.fc_out = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        # Passaggio nei livelli convoluzionali con attivazione ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Appiattimento del tensore spaziale in un vettore 1D per i livelli lineari
        x = x.view(x.size(0), -1) 
        
        # Passaggio nei livelli lineari
        x = F.relu(self.fc1(x))
        
        # OUTPUT LAYER: Restituiamo i "logits" crudi.
        # NESSUNA ATTIVAZIONE QUI! La CrossEntropyLoss di PyTorch vuole questi valori.
        logits = self.fc_out(x)
        
        return logits
    
    def predict_probabilities(self, x):
        """
        Funzione helper da usare SOLO in inferenza (server.py / inference.py)
        quando vogliamo estrarre le percentuali reali per il Filtro di Sicurezza.
        """
        # Disabilita il calcolo dei gradienti per risparmiare VRAM e CPU
        with torch.no_grad():
            logits = self.forward(x)
            # Applica il Softmax sulla dimensione delle classi (dim=1)
            probabilities = F.softmax(logits, dim=1)
        return probabilities