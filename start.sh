#!/bin/bash

# 1. Attiva il tuo ambiente conda (sostituisci 'torchfish' con il nome del tuo ambiente se ne usi uno già esistente)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchfish

echo "Avvio dell'API FastAPI in background..."
# 2. Avvia il server Python. La '&' finale manda il processo in background senza bloccare il terminale
python server.py &
# Salviamo l'ID del processo (PID) del server per poterlo chiudere dopo
API_PID=$!

echo "Avvio dell'interfaccia React..."
# 3. Entra nella cartella frontend e avvia Vite
cd torchfish-ui
npm run dev &
FRONTEND_PID=$!

# 4. Funzione di "pulizia" che si attiva quando decidi di uscire
cleanup() {
    echo -e "\nChiusura di TorchFish in corso..."
    kill $API_PID
    kill $FRONTEND_PID
    echo "Processi terminati."
    exit
}

# 5. Mettiti in ascolto del segnale SIGINT (quello che invii premendo Ctrl + C nel terminale)
trap cleanup SIGINT

# 6. Mantiene lo script attivo in attesa del tuo Ctrl + C
wait