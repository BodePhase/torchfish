# TorchFish

## 1. Descrizione del progetto

**TorchFish** è un motore scacchistico ibrido e un'applicazione web full-stack progettata per riprodurre lo stile di gioco dei Grandi Maestri (a partire da Garry Kasparov) combinando Deep Learning e ricerca tattica classica. Il sistema unisce l'intuito posizionale di una Policy Network convoluzionale in PyTorch con il calcolo rigoroso di un motore Alpha-Beta (Negamax con Quiescence Search e valutazione PeSTO Tapered). L'applicazione include un'interfaccia grafica reattiva in stile Lichess/Chess.com con telemetria neurale in tempo reale, barra di valutazione e selettore di profili storici.

---

## 2. Requisiti e installazione

### Requisiti di sistema
* **Python:** versione 3.10+ (consigliato il supporto CUDA per l'inferenza e il training su GPU PyTorch).
* **Librerie Python:** `torch`, `numpy`, `chess` (`python-chess`), `fastapi`, `uvicorn`, `tqdm`.
* **Node.js e npm:** Node.js 18+ per l'esecuzione del frontend React/Vite (`torchfish-ui`).

### Guida all'installazione e all'avvio

1. **Clonare il repository:**
   ```bash
   git clone [DA COMPLETARE: URL_REPOSITORY_GITHUB]
   cd torchfish
   ```

2. **Configurare l'ambiente virtuale Python e installare le dipendenze del backend:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install fastapi uvicorn tqdm chess numpy torch
   ```

3. **(Opzionale) Generare il dataset tattico anti-blunder ed eseguire il fine-tuning della rete:**
   ```bash
   python augment_tactics.py
   python finetune_policy.py
   ```

4. **Avviare il server backend (FastAPI + Motore TorchFish):**
   ```bash
   python server.py
   ```
   Il backend resterà in ascolto all'indirizzo `http://localhost:8000`.

5. **Installare le dipendenze e avviare l'interfaccia web React:**
   Aprire un secondo terminale nella radice del progetto ed eseguire:
   ```bash
   cd torchfish-ui
   npm install
   npm run dev
   ```

### Esempio di utilizzo
Una volta avviati entrambi i servizi, aprire il browser all'URL mostrato da Vite (solitamente `http://localhost:5173`). Dalla barra laterale destra è possibile selezionare il profilo del Grande Maestro avversario (*Kasparov*, *Tal*, *Karpov*, *Carlsen*), attivare la visualizzazione delle frecce candidate dell'IA sulla scacchiera (`🎯 Frecce IA`), navigare nello storico delle mosse tramite tastiera o pulsanti dedicati ed esportare la partita in formato `FEN` o `PGN`.

---

## 3. Caratteristiche tecniche

### Architettura Ibrida: Policy Network e Ricerca Tattica
Una Policy Network pura addestrata per *Behavioral Cloning* predice le mosse sulla base di pattern geometrici ma è priva di calcolo concreto (*0-ply lookahead*), risultando vulnerabile a sviste tattiche in posizioni fuori distribuzione. TorchFish risolve questo limite integrando due livelli:
* **`KasparovPolicyNet` (PyTorch):** Rete neurale convoluzionale che mappa lo stato della scacchiera in una distribuzione di probabilità Softmax su un dizionario di **4544 mosse UCI** (`models/uci_dictionary.json`), fornendo i *prior* posizionali alla radice.
* **Synthetic Blunder Injection (`augment_tactics.py`):** Pipeline di data augmentation che arricchisce le 81.700 posizioni storiche di Kasparov con 25.000 posizioni tattiche sintetiche (matti in 1 e catture di pezzi lasciati indifesi), seguita da un fine-tuning mirato (`finetune_policy.py`).
* **Meccanismo *Blunder Lock*:** Alla radice, i bonus stilistici della rete neurale e dell'Opening Book vengono applicati solo alle mosse tatticamente sane; se una mossa perde $\ge 85$ centipawn rispetto alla migliore variante calcolata dal motore Alpha-Beta, i bonus di stile vengono azzerati per impedire errori materiali.

### Algoritmo di ricerca
Il motore tattico (`inference.py`) implementa un albero di ricerca **Iterative Deepening Negamax** con le seguenti tecniche di potatura ed estensione:
* **Principal Variation Search (PVS) & Alpha-Beta Pruning:** Esplora la variante principale a finestra piena e utilizza finestre ridotte (*Null Window*) per confutare rapidamente le alternative sub-ottimali.
* **Quiescence Search con Delta Pruning:** Estende la ricerca oltre le foglie dell'albero principale valutando tutte le sequenze di catture, promozioni ed evasioni dallo scacco (fino a 6-7 ply aggiuntivi) per eliminare l'effetto orizzonte.
* **Null Move Pruning (NMP):** Nelle posizioni non sotto scacco e con materiale non pedonale sufficiente (`depth >= 3`), verifica se concedere una mossa vuota all'avversario mantenga comunque il punteggio sopra `beta`, tagliando i rami ridondanti.
* **Late Move Reductions (LMR):** Riduce di 1 o 2 ply la profondità di analisi per le mosse silenziose ordinate in fondo alla lista, ricalcolandole a profondità piena solo se superano `alpha`.
* **Check Extensions:** Estende automaticamente di 1 ply la profondità quando il giocatore di turno si trova sotto scacco, evitando che scacchi di distrazione nascondano perdite materiali oltre l'orizzonte di ricerca.

### Sistema di valutazione (PeSTO Tapered Evaluation)
La funzione di valutazione statica (`_static_eval_cp`) opera in centesimi di pedone (*centipawns*) sfruttando i bitboard nativi di `python-chess`:

| Componente | Descrizione tecnica | Scopo scacchistico |
| :--- | :--- | :--- |
| **Tapered Eval (PeSTO)** | Interpolazione lineare tra punteggio di Mediogioco (`MG`) e Finale (`EG`) basata su `game_phase` (da `24` a `0`). | Consente al Re di restare protetto nell'arrocco durante il mediogioco e di centralizzarsi attivamente nei finali. |
| **Tabelle PST Pre-fuse** | Matrici `MG_WHITE/BLACK` ed `EG_WHITE/BLACK` con valori materiali e posizionali pre-sommati per tutte le 64 case. | Esegue la valutazione materiale e posizionale in $O(1)$ per pezzo senza moltiplicazioni a runtime. |
| **Bitboard Passed Pawns** | Maschere binarie a 64 bit pre-calcolate (`PASSED_MASK_WHITE/BLACK`) confrontate tramite `AND` bit-a-bit. | Riconosce istantaneamente i pedoni passati assegnando bonus crescenti fino a `+215 cp` in settima traversa. |
| ** Struttura e Pezzi** | Bonus per la coppia degli Alfieri (`+30 MG / +45 EG`) e per le Torri su colonne aperte (`+22 cp`) o semi-aperte (`+10 cp`). | Premia il controllo delle linee aperte e il coordinamento dei pezzi a lungo raggio. |
| **Sicurezza Arrocco** | Penalità nel mediogioco (`-70 cp` nella statica, `-140 cp` alla radice) per la perdita artificiale dei diritti d'arrocco. | Impedisce mosse premature di Re o Torre prima dello sviluppo completo. |
| **Mop-Up Evaluation** | Attiva per `game_phase <= 6` e vantaggio $> 300$ cp tramite distanza dal centro (`CENTER_DIST`) e distanza tra i Re. | Spinge il Re avversario verso il bordo e avvicina il Re vincente per forzare lo scacco matto nei finali. |

### Ottimizzazioni, Ordinamento Mosse e Gestione del Tempo
* **Ordinamento Mosse (`_order_moves`):** Ordina i nodi dando priorità a: 1) Mossa della Transposition Table (`100.000`), 2) Catture ordinate per *MVV-LVA* (*Most Valuable Victim - Least Valuable Attacker*), 3) Promozioni, 4) *Killer Moves* (2 mosse silenziose per ply che hanno causato un beta-cutoff), 5) Disimpegno di pezzi attaccati.
* **Transposition Table (`self.tt`):** Tabella hash fino a 200.000 posizioni con flag `TT_EXACT`, `TT_LOWER` e `TT_UPPER` per riutilizzare i sotto-alberi già valutati tra le iterazioni dell'Iterative Deepening.
* **Controllo del Tempo (*Hard-Stop*):** Budget predefinito di `600 ms` per mossa, monitorato ogni 1024 nodi tramite eccezione dedicata `SearchTimeout`, garantendo il completamento minimo del 2° ply e la restituzione immediata della miglior mossa trovata all'iterazione completa più profonda.
* **Memoria Anti-Ripetizione e Opening Book:** Coda circolare (`deque`) delle ultime 16 posizioni per penalizzare il pendolarismo sterile e dizionario delle aperture estratto dai file CSV delle partite storiche.

### Protocollo API e Interfaccia Web (`torchfish-ui`)
Il motore non utilizza il protocollo testuale UCI su standard input/output, ma espone una **REST API JSON** tramite **FastAPI** (`server.py`):
* `POST /api/evaluate`: Riceve il `fen`, il profilo `legend` e restituisce la mossa scelta (`best_move_san`, `best_move_uci`), la valutazione in pedoni dal punto di vista del Bianco (`eval_white`), l'eventuale contatore di matto (`mate_in`), la profondità raggiunta e la `top_moves` con le probabilità normalizzate.
* `POST /api/eval_only`: Valutazione rapida di quiescenza per aggiornare la barra di valutazione subito dopo la mossa dell'utente.
* `GET /api/legends`: Restituisce i metadati dei profili disponibili (*Kasparov*, *Tal*, *Karpov*, *Carlsen*).
* **Frontend React (`App.jsx`):** Gestisce lo stato della partita tramite un array di snapshot immutabili per consentire il *time-travel* istantaneo $O(1)$ nello storico mosse, integrando `AbortController` per annullare le richieste pendenti, selettore grafico di promozione, evidenziazione delle case legali, frecce tattiche disegnabili col tasto destro e riproduzione degli effetti sonori ufficiali di Lichess.
