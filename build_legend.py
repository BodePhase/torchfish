import argparse
import csv
import glob
import json
import os
import random
import re
import time
import chess
import chess.pgn
import torch
import torch.nn as nn
from tqdm import tqdm
from models.policy_net import KasparovPolicyNet

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Configurazione percorsi e regex con word-boundary per ciascuna Leggenda
LEGENDS_META = {
    "kasparov": {
        "name": "Garry Kasparov",
        "regex": re.compile(r"\bkasparov\b", re.IGNORECASE),
        "dir": "data/gm_kasparov",
        "default_pgn": "data/gm_kasparov/Kasparov.pgn",
        "book_csv": "data/gm_kasparov/kasparov_moves.csv",
        "aug_csv": "data/gm_kasparov/kasparov_tactical_augmented.csv",
        "weights": "weights/kasparov_policy.pth",
    },
    "tal": {
        "name": "Mikhail Tal",
        "regex": re.compile(r"\btal\b", re.IGNORECASE),
        "dir": "data/gm_tal",
        "default_pgn": "data/gm_tal/Tal.pgn",
        "book_csv": "data/gm_tal/tal_moves.csv",
        "aug_csv": "data/gm_tal/tal_tactical_augmented.csv",
        "weights": "weights/tal_policy.pth",
    },
    "karpov": {
        "name": "Anatoly Karpov",
        "regex": re.compile(r"\bkarpov\b", re.IGNORECASE),
        "dir": "data/gm_karpov",
        "default_pgn": "data/gm_karpov/Karpov.pgn",
        "book_csv": "data/gm_karpov/karpov_moves.csv",
        "aug_csv": "data/gm_karpov/karpov_tactical_augmented.csv",
        "weights": "weights/karpov_policy.pth",
    },
    "carlsen": {
        "name": "Magnus Carlsen",
        "regex": re.compile(r"\bcarlsen\b", re.IGNORECASE),
        "dir": "data/gm_carlsen",
        "default_pgn": "data/gm_carlsen/Carlsen.pgn",
        "book_csv": "data/gm_carlsen/carlsen_moves.csv",
        "aug_csv": "data/gm_carlsen/carlsen_tactical_augmented.csv",
        "weights": "weights/carlsen_policy.pth",
    },
}


def resolve_pgn_path(meta: dict, custom_pgn: str = None) -> str:
    """Trova il file PGN specificato o cerca automaticamente un .pgn nella cartella della Leggenda."""
    if custom_pgn:
        if not os.path.exists(custom_pgn):
            raise FileNotFoundError(f"File PGN specificato non trovato: {custom_pgn}")
        return custom_pgn

    if os.path.exists(meta["default_pgn"]):
        return meta["default_pgn"]

    found = glob.glob(os.path.join(meta["dir"], "*.pgn"))
    if found:
        return found[0]

    raise FileNotFoundError(
        f"Nessun file .pgn trovato in '{meta['dir']}/'. "
        f"Inserisci '{meta['default_pgn']}' oppure passa --pgn <percorso_file.pgn>."
    )


def load_valid_uci_set(dict_path: str) -> set:
    with open(dict_path, "r", encoding="utf-8") as f:
        raw_dict = json.load(f)
    if "move_to_int" in raw_dict:
        return set(raw_dict["move_to_int"].keys())
    return set(raw_dict.keys())


def step1_extract_pgn_to_csv(pgn_path: str, output_csv: str, player_regex: re.Pattern, valid_uci: set) -> int:
    """Estrae tutte le mosse della Leggenda dal file PGN e genera il CSV per l'Opening Book."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []
    games_matched = 0
    games_total = 0

    print(f"\n[Step 1/3] Estrazione partite da '{pgn_path}'...")
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games_total += 1

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            is_white = bool(player_regex.search(white))
            is_black = bool(player_regex.search(black))

            if not (is_white or is_black):
                continue

            games_matched += 1
            board = game.board()
            for move in game.mainline_moves():
                if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and is_black):
                    uci = move.uci()
                    if uci in valid_uci:
                        rows.append({"fen": board.fen(), "move_uci": uci})
                board.push(move)

    if not rows:
        raise ValueError(
            f"Nessuna partita trovata per il pattern '{player_regex.pattern}' su {games_total} partite lette."
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> Partite trovate: {games_matched}/{games_total} | Mosse estratte: {len(rows)}")
    print(f"  -> Opening Book salvato in: {output_csv}")
    return len(rows)


def find_tactical_punishment(board: chess.Board, valid_uci: set):
    """Individua un Matto in 1 o la migliore cattura netta (>= +280 cp) sulla scacchiera."""
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        uci = move.uci()
        if uci not in valid_uci:
            continue
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return uci, "mate_in_1"

    best_capture = None
    best_gain = 0

    for move in legal_moves:
        if not board.is_capture(move):
            continue
        uci = move.uci()
        if uci not in valid_uci:
            continue

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if not victim or not attacker:
            continue

        victim_val = PIECE_VALUES.get(victim.piece_type, 0)
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0)

        board.push(move)
        is_recaptured = board.is_attacked_by(board.turn, move.to_square)
        board.pop()

        net_gain = victim_val if not is_recaptured else (victim_val - attacker_val)
        if net_gain >= 280 and net_gain > best_gain:
            best_gain = net_gain
            best_capture = uci

    if best_capture:
        return best_capture, "hanging_piece"
    return None, None


def step2_augment_tactics(
    input_csv: str,
    output_csv: str,
    valid_uci: set,
    tactical_ratio: float = 0.20
) -> int:
    """Inietta posizioni tattiche sintetiche (Matto in 1 e pezzi in presa) nel dataset della Leggenda."""
    with open(input_csv, "r", encoding="utf-8") as f:
        original_rows = list(csv.DictReader(f))

    target_tactical = int(len(original_rows) * tactical_ratio)
    print(f"\n[Step 2/3] Iniezione di ~{target_tactical} posizioni tattiche anti-blunder...")

    sampled_rows = list(original_rows)
    random.shuffle(sampled_rows)

    tactical_rows = []
    stats = {"mate_in_1": 0, "hanging_piece": 0}

    for row in tqdm(sampled_rows, desc="Augmentation tattica"):
        if len(tactical_rows) >= target_tactical:
            break
        try:
            board = chess.Board(row["fen"])
            if board.is_game_over():
                continue

            candidate_blunders = [
                m for m in board.legal_moves
                if board.piece_type_at(m.from_square) in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
            ]
            if not candidate_blunders:
                continue

            blunder_move = random.choice(candidate_blunders)
            board.push(blunder_move)
            if board.is_game_over():
                continue

            punish_uci, reason = find_tactical_punishment(board, valid_uci)
            if punish_uci:
                tactical_rows.append({"fen": board.fen(), "move_uci": punish_uci})
                stats[reason] += 1
        except Exception:
            continue

    combined_rows = original_rows + tactical_rows
    random.shuffle(combined_rows)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci"])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(
        f"  -> Tattiche generate: {len(tactical_rows)} "
        f"(Matti in 1: {stats['mate_in_1']}, Pezzi in presa: {stats['hanging_piece']})"
    )
    print(f"  -> Dataset completo salvato in: {output_csv} ({len(combined_rows)} posizioni)")
    return len(combined_rows)


_FEN_CHAR_TO_CH = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def _load_dataset_uint8(csv_path: str, dict_path: str):
    """
    Codifica il CSV una sola volta in RAM come tensore uint8 [N, 14, 8, 8] identico
    a dataset_policy.board_to_tensor, evitando os.fork() dopo l'init CUDA in WSL2.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        move_to_int = json.load(f)["move_to_int"]

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    boards_u8 = torch.zeros((n, 14, 8, 8), dtype=torch.uint8)
    targets = torch.zeros(n, dtype=torch.long)
    b_np = boards_u8.numpy()
    t_np = targets.numpy()

    for i, row in enumerate(tqdm(rows, desc="Pre-codifica tensori in RAM", leave=False)):
        parts = row["fen"].split()
        ranks = parts[0].split("/")
        for r_idx, rank_str in enumerate(ranks):
            board_row = 7 - r_idx  # Traversa 1 = row 0, Traversa 8 = row 7
            col = 0
            for ch in rank_str:
                if ch.isdigit():
                    col += int(ch)
                else:
                    b_np[i, _FEN_CHAR_TO_CH[ch], board_row, col] = 1
                    col += 1
        if len(parts) > 1 and parts[1] == "w":
            b_np[i, 12, :, :] = 1

        t_np[i] = move_to_int[row["move_uci"]]

    return boards_u8, targets, len(move_to_int)


def step3_train_policy(
    csv_path: str,
    dict_path: str,
    output_weights: str,
    finetune_from: str,
    epochs: int,
    batch_size: int,
    lr: float,
):
    """Addestra da zero (o esegue fine-tuning) KasparovPolicyNet direttamente da RAM."""
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    # 1. Pre-codifica in RAM su singolo processo PRIMA di inizializzare il contesto CUDA
    print(f"\n[Step 3/3] Caricamento dataset '{csv_path}' in RAM...")
    boards_u8, targets_cpu, num_classes = _load_dataset_uint8(csv_path, dict_path)
    num_samples = boards_u8.size(0)
    ram_mb = boards_u8.numel() / (1024 * 1024)
    print(f"  -> {num_samples} posizioni pronte in RAM ({ram_mb:.1f} MB uint8).")

    # 2. Inizializza CUDA solo dopo aver preparato i tensori in RAM (zero fork/IPC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = KasparovPolicyNet(num_classes=num_classes).to(device)
    except TypeError:
        model = KasparovPolicyNet().to(device)

    if finetune_from and os.path.exists(finetune_from):
        print(f"  -> Fine-tuning dai pesi '{finetune_from}' su {device}...")
        ckpt = torch.load(finetune_from, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"  -> Training dedicato da zero (from scratch) su {device}...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        correct = 0
        t0 = time.perf_counter()

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = perm[start_idx : start_idx + batch_size]
            boards = boards_u8[batch_idx].to(device=device, dtype=torch.float32)
            targets = targets_cpu[batch_idx].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(boards)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bsz = boards.size(0)
            total_loss += loss.item() * bsz
            correct += (logits.argmax(dim=-1) == targets).sum().item()

        scheduler.step()
        avg_loss = total_loss / max(1, num_samples)
        acc = (correct / max(1, num_samples)) * 100.0
        elapsed = time.perf_counter() - t0
        print(f"  Epoca {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Tempo: {elapsed:.2f}s")

    torch.save(model.state_dict(), output_weights)
    print(f"\n[Completato] Pesi neurali salvati in: {output_weights}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline unificata per generare Opening Book e Pesi Policy da zero per ciascuna Leggenda."
    )
    parser.add_argument(
        "--legend",
        type=str,
        required=True,
        choices=list(LEGENDS_META.keys()),
        help="ID della leggenda da addestrare (kasparov, tal, karpov, carlsen)",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="Percorso opzionale al file .pgn (default: cerca automaticamente in data/gm_<legend>/)",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="models/uci_dictionary.json",
        help="Percorso al dizionario UCI JSON",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Opzionale: percorso a un file .pth se si desidera fare fine-tuning invece di addestrare da zero",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Numero di epoche (default: 25)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Dimensione del batch (default: 1024)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate iniziale (default: 1e-3)")
    parser.add_argument(
        "--tactical-ratio",
        type=float,
        default=0.20,
        help="Rapporto di posizioni tattiche sintetiche da iniettare (default: 0.20)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Salta gli Step 1 e 2 se il CSV aumentato è già presente",
    )

    args = parser.parse_args()
    meta = LEGENDS_META[args.legend]

    print(f"=== TorchFish Legend Builder: {meta['name']} ({args.legend}) ===")
    valid_uci = load_valid_uci_set(args.dict)

    if not args.skip_extract:
        pgn_file = resolve_pgn_path(meta, args.pgn)
        step1_extract_pgn_to_csv(
            pgn_path=pgn_file,
            output_csv=meta["book_csv"],
            player_regex=meta["regex"],
            valid_uci=valid_uci,
        )
        step2_augment_tactics(
            input_csv=meta["book_csv"],
            output_csv=meta["aug_csv"],
            valid_uci=valid_uci,
            tactical_ratio=args.tactical_ratio,
        )
    elif not os.path.exists(meta["aug_csv"]):
        raise FileNotFoundError(f"Impossibile usare --skip-extract: {meta['aug_csv']} non esiste.")

    step3_train_policy(
        csv_path=meta["aug_csv"],
        dict_path=args.dict,
        output_weights=meta["weights"],
        finetune_from=args.finetune_from,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()