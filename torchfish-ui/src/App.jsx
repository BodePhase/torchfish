import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import './App.css';

const INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const PIECE_SYMBOLS = {
  w: { p: '♙', n: '♘', b: '♗', r: '♖', q: '♕', k: '♔' },
  b: { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' }
};

const PROMOTION_PIECES = ['q', 'r', 'b', 'n'];

// Suoni ufficiali Lichess (repository open-source lila)
const SOUND_URLS = {
  move: 'https://raw.githubusercontent.com/ornicar/lila/master/public/sound/standard/Move.mp3',
  capture: 'https://raw.githubusercontent.com/ornicar/lila/master/public/sound/standard/Capture.mp3',
  check: 'https://raw.githubusercontent.com/ornicar/lila/master/public/sound/standard/Check.mp3',
  castle: 'https://raw.githubusercontent.com/ornicar/lila/master/public/sound/standard/Castle.mp3',
  gameEnd: 'https://raw.githubusercontent.com/ornicar/lila/master/public/sound/standard/GenericNotify.mp3',
};

const calculateMaterial = (chessInstance) => {
  const startingPieces = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const currentPieces = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0 }
  };
  const values = { p: 1, n: 3, b: 3, r: 5, q: 9 };

  chessInstance.board().forEach((row) => {
    row.forEach((piece) => {
      if (piece && piece.type !== 'k') {
        currentPieces[piece.color][piece.type]++;
      }
    });
  });

  const getCaptured = (color) => {
    const captured = [];
    let score = 0;
    for (const type in startingPieces) {
      const diff = Math.max(0, startingPieces[type] - currentPieces[color][type]);
      for (let i = 0; i < diff; i++) {
        captured.push(PIECE_SYMBOLS[color][type]);
        score += values[type];
      }
    }
    return { captured, score };
  };

  const white = getCaptured('w');
  const black = getCaptured('b');

  return {
    whiteCaptured: white.captured,
    blackCaptured: black.captured,
    whiteAdvantage: white.score < black.score ? black.score - white.score : 0,
    blackAdvantage: black.score < white.score ? white.score - black.score : 0,
  };
};

const formatMoveToFigurine = (san) => {
  if (!san) return '';
  return san
    .replace(/N/g, '♘')
    .replace(/B/g, '♗')
    .replace(/R/g, '♖')
    .replace(/Q/g, '♕')
    .replace(/K/g, '♔');
};

const findCheckedKingSquare = (chessInstance) => {
  const inCheck = typeof chessInstance.inCheck === 'function'
    ? chessInstance.inCheck()
    : chessInstance.in_check?.();
  if (!inCheck) return null;

  const turn = chessInstance.turn();
  const board = chessInstance.board();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece && piece.type === 'k' && piece.color === turn) {
        return piece.square;
      }
    }
  }
  return null;
};

const detectGameOver = (chessInstance) => {
  const isOver = typeof chessInstance.isGameOver === 'function'
    ? chessInstance.isGameOver()
    : chessInstance.game_over?.();
  if (!isOver) return null;

  const isCheckmate = chessInstance.isCheckmate?.() ?? chessInstance.in_checkmate?.();
  if (isCheckmate) {
    const winner = chessInstance.turn() === 'w' ? 'Nero (Kasparov-Net)' : 'Bianco';
    return { title: 'Scacco Matto!', subtitle: `Vittoria del ${winner}` };
  }
  if (chessInstance.isStalemate?.() ?? chessInstance.in_stalemate?.()) {
    return { title: 'Patta per Stallo', subtitle: 'Nessuna mossa legale disponibile' };
  }
  if (chessInstance.isThreefoldRepetition?.() ?? chessInstance.in_threefold_repetition?.()) {
    return { title: 'Patta', subtitle: 'Tripla ripetizione di posizione' };
  }
  if (chessInstance.isInsufficientMaterial?.() ?? chessInstance.insufficient_material?.()) {
    return { title: 'Patta', subtitle: 'Materiale insufficiente' };
  }
  return { title: 'Patta', subtitle: 'Regola delle 50 mosse' };
};

export default function App() {
  const masterGameRef = useRef(new Chess());
  const abortControllerRef = useRef(null);
  const moveListRef = useRef(null);
  const audioCacheRef = useRef({});

  const [snapshots, setSnapshots] = useState([
    { fen: INITIAL_FEN, san: null, from: null, to: null }
  ]);
  const [viewIndex, setViewIndex] = useState(0);

  const [playerColor, setPlayerColor] = useState('w');
  const [kasparovThinking, setKasparovThinking] = useState(false);
  const [moveFrom, setMoveFrom] = useState(null);
  const [rightClickedSquares, setRightClickedSquares] = useState({});
  const [pendingPromotion, setPendingPromotion] = useState(null);
  const [resignedWinner, setResignedWinner] = useState(null);
  const [telemetry, setTelemetry] = useState({ confidence: null, topMoves: [] });
  const [copiedBadge, setCopiedBadge] = useState(null);

  // Nuovi stati per Audio e Frecce IA
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [showAIArrows, setShowAIArrows] = useState(false);

  // Pre-caricamento degli effetti sonori di Lichess
  useEffect(() => {
    Object.entries(SOUND_URLS).forEach(([key, url]) => {
      const audio = new Audio(url);
      audio.preload = 'auto';
      audio.volume = 0.75;
      audioCacheRef.current[key] = audio;
    });
  }, []);

  // Riproduce il suono appropriato in base al contesto della mossa
  const playMoveSound = useCallback((moveObj, chessInstance) => {
    if (!soundEnabled || !moveObj) return;

    const isOver = chessInstance.isGameOver?.() ?? chessInstance.game_over?.();
    const inCheck = chessInstance.inCheck?.() ?? chessInstance.in_check?.();

    let soundKey = 'move';
    if (isOver) {
      soundKey = 'gameEnd';
    } else if (inCheck) {
      soundKey = 'check';
    } else if (moveObj.captured || moveObj.san?.includes('x')) {
      soundKey = 'capture';
    } else if (moveObj.san?.includes('O-O')) {
      soundKey = 'castle';
    }

    const audio = audioCacheRef.current[soundKey];
    if (audio) {
      audio.currentTime = 0;
      audio.play().catch(() => {});
    }
  }, [soundEnabled]);

  const isViewingHistory = viewIndex < snapshots.length - 1;
  const currentSnapshot = snapshots[viewIndex];

  const viewedGame = useMemo(() => new Chess(currentSnapshot.fen), [currentSnapshot.fen]);
  const material = useMemo(() => calculateMaterial(viewedGame), [viewedGame]);

  const gameOverStatus = useMemo(() => {
    if (resignedWinner) {
      return { title: 'Partita Terminata', subtitle: `Abbandono — Vince ${resignedWinner}` };
    }
    const liveGame = new Chess(snapshots[snapshots.length - 1].fen);
    return detectGameOver(liveGame);
  }, [snapshots, resignedWinner]);

  const legalMovesForSelected = useMemo(() => {
    if (!moveFrom || isViewingHistory || gameOverStatus) return [];
    return viewedGame.moves({ square: moveFrom, verbose: true });
  }, [viewedGame, moveFrom, isViewingHistory, gameOverStatus]);

  useEffect(() => {
    if (moveListRef.current && viewIndex === snapshots.length - 1) {
      moveListRef.current.scrollTop = moveListRef.current.scrollHeight;
    }
  }, [snapshots.length, viewIndex]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setViewIndex((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setViewIndex((prev) => {
          const nextIdx = Math.min(snapshots.length - 1, prev + 1);
          if (nextIdx !== prev && snapshots[nextIdx]) {
            const tempGame = new Chess(snapshots[nextIdx].fen);
            playMoveSound(snapshots[nextIdx], tempGame);
          }
          return nextIdx;
        });
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setViewIndex(0);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setViewIndex(snapshots.length - 1);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [snapshots, playMoveSound]);

  const makeAIMove = useCallback(async (currentFen) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setKasparovThinking(true);
    try {
      const response = await fetch('http://localhost:8000/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: currentFen }),
        signal: controller.signal,
      });

      const data = await response.json();
      await new Promise((r) => setTimeout(r, 450));

      if (controller.signal.aborted) return;

      if (data.best_move_san) {
        const moveObj = masterGameRef.current.move(data.best_move_san);
        if (moveObj) {
          const newFen = masterGameRef.current.fen();
          playMoveSound(moveObj, masterGameRef.current);

          setSnapshots((prev) => {
            const next = [
              ...prev,
              { fen: newFen, san: moveObj.san, from: moveObj.from, to: moveObj.to, captured: moveObj.captured }
            ];
            setViewIndex(next.length - 1);
            return next;
          });
          setTelemetry({
            confidence: data.confidence ?? null,
            topMoves: data.top_moves ?? [{ san: data.best_move_san, uci: data.best_move_uci, prob: data.confidence ?? 100 }]
          });
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Errore API:', error);
      }
    } finally {
      if (!controller.signal.aborted) {
        setKasparovThinking(false);
      }
    }
  }, [playMoveSound]);

  const commitUserMove = useCallback((from, to, promotionPiece = 'q') => {
    try {
      const moveObj = masterGameRef.current.move({
        from,
        to,
        promotion: promotionPiece,
      });

      if (!moveObj) return false;

      const newFen = masterGameRef.current.fen();
      playMoveSound(moveObj, masterGameRef.current);

      setSnapshots((prev) => {
        const next = [
          ...prev,
          { fen: newFen, san: moveObj.san, from: moveObj.from, to: moveObj.to, captured: moveObj.captured }
        ];
        setViewIndex(next.length - 1);
        return next;
      });

      setMoveFrom(null);
      setPendingPromotion(null);
      setRightClickedSquares({});

      const isOver = masterGameRef.current.isGameOver?.() ?? masterGameRef.current.game_over?.();
      if (!isOver) {
        setTimeout(() => makeAIMove(newFen), 50);
      }
      return true;
    } catch {
      return false;
    }
  }, [makeAIMove, playMoveSound]);

  const isPromotionMove = (from, to) => {
    const piece = masterGameRef.current.get(from);
    if (!piece || piece.type !== 'p' || piece.color !== playerColor) return false;
    const isLastRank = (piece.color === 'w' && to[1] === '8') || (piece.color === 'b' && to[1] === '1');
    if (!isLastRank) return false;

    const legal = masterGameRef.current.moves({ square: from, verbose: true });
    return legal.some((m) => m.to === to);
  };

  function onPieceDragBegin(piece, sourceSquare) {
    if (kasparovThinking || isViewingHistory || gameOverStatus) return;
    if (piece.charAt(0) === playerColor && masterGameRef.current.turn() === playerColor) {
      setMoveFrom(sourceSquare);
    }
    setRightClickedSquares({});
  }

  function onDrop(sourceSquare, targetSquare) {
    if (kasparovThinking || isViewingHistory || gameOverStatus) return false;
    if (masterGameRef.current.turn() !== playerColor) return false;

    if (isPromotionMove(sourceSquare, targetSquare)) {
      setPendingPromotion({ from: sourceSquare, to: targetSquare, color: playerColor });
      return false;
    }

    const success = commitUserMove(sourceSquare, targetSquare);
    if (!success) setMoveFrom(null);
    return success;
  }

  function onSquareClick(square) {
    if (kasparovThinking || isViewingHistory || gameOverStatus) return;
    if (masterGameRef.current.turn() !== playerColor) return;

    setRightClickedSquares({});

    if (!moveFrom) {
      const hasPiece = masterGameRef.current.get(square);
      if (hasPiece && hasPiece.color === playerColor) {
        setMoveFrom(square);
      }
      return;
    }

    if (moveFrom === square) {
      setMoveFrom(null);
      return;
    }

    if (isPromotionMove(moveFrom, square)) {
      setPendingPromotion({ from: moveFrom, to: square, color: playerColor });
      return;
    }

    const success = commitUserMove(moveFrom, square);
    if (!success) {
      const hasPiece = masterGameRef.current.get(square);
      if (hasPiece && hasPiece.color === playerColor) {
        setMoveFrom(square);
      } else {
        setMoveFrom(null);
      }
    }
  }

  function onSquareRightClick(square) {
    setRightClickedSquares((prev) => {
      const next = { ...prev };
      if (next[square]) delete next[square];
      else next[square] = { backgroundColor: 'rgba(235, 97, 80, 0.8)' };
      return next;
    });
  }

  const startNewGame = (color = playerColor) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    masterGameRef.current = new Chess();
    setSnapshots([{ fen: INITIAL_FEN, san: null, from: null, to: null }]);
    setViewIndex(0);
    setPlayerColor(color);
    setKasparovThinking(false);
    setMoveFrom(null);
    setPendingPromotion(null);
    setRightClickedSquares({});
    setResignedWinner(null);
    setTelemetry({ confidence: null, topMoves: [] });

    if (color === 'b') {
      setTimeout(() => makeAIMove(INITIAL_FEN), 150);
    }
  };

  const handleFlipColor = () => {
    const nextColor = playerColor === 'w' ? 'b' : 'w';
    startNewGame(nextColor);
  };

  const handleResign = () => {
    if (gameOverStatus) return;
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setKasparovThinking(false);
    setResignedWinner('TorchFish (Kasparov)');
    if (soundEnabled && audioCacheRef.current.gameEnd) {
      audioCacheRef.current.gameEnd.currentTime = 0;
      audioCacheRef.current.gameEnd.play().catch(() => {});
    }
  };

  const copyToClipboard = (type) => {
    const text = type === 'FEN' ? currentSnapshot.fen : masterGameRef.current.pgn();
    navigator.clipboard.writeText(text || INITIAL_FEN);
    setCopiedBadge(type);
    setTimeout(() => setCopiedBadge(null), 1800);
  };

  const movePairs = useMemo(() => {
    const pairs = [];
    for (let i = 1; i < snapshots.length; i += 2) {
      pairs.push({
        moveNumber: Math.ceil(i / 2),
        white: { ...snapshots[i], index: i },
        black: snapshots[i + 1] ? { ...snapshots[i + 1], index: i + 1 } : null,
      });
    }
    return pairs;
  }, [snapshots]);

  // Frecce candidate dell'IA (attive solo se l'utente abilita "Frecce IA", così non sovrascrivono il drag destro manuale)
  const aiCandidateArrows = useMemo(() => {
    if (!showAIArrows || !telemetry.topMoves.length) return undefined;
    const colors = [
      'rgba(32, 110, 167, 0.85)', // 1° scelta: Blu Lichess
      'rgba(98, 153, 36, 0.70)',  // 2° scelta: Verde
      'rgba(230, 157, 0, 0.55)'   // 3° scelta: Giallo/Arancio
    ];
    return telemetry.topMoves
      .filter((m) => m.uci && m.uci.length >= 4)
      .map((m, idx) => [m.uci.slice(0, 2), m.uci.slice(2, 4), colors[idx] || colors[2]]);
  }, [showAIArrows, telemetry.topMoves]);

  const customSquareStyles = useMemo(() => {
    const styles = {};

    if (currentSnapshot.from && currentSnapshot.to) {
      styles[currentSnapshot.from] = { backgroundColor: 'rgba(244, 246, 128, 0.6)' };
      styles[currentSnapshot.to] = { backgroundColor: 'rgba(244, 246, 128, 0.6)' };
    }

    if (moveFrom && !isViewingHistory) {
      styles[moveFrom] = { backgroundColor: 'rgba(244, 246, 128, 0.65)' };
    }

    legalMovesForSelected.forEach((m) => {
      const isCapture = Boolean(m.captured) || m.flags.includes('e');
      styles[m.to] = {
        ...styles[m.to],
        background: isCapture
          ? 'radial-gradient(transparent 0%, transparent 78%, rgba(20, 85, 30, 0.35) 80%)'
          : 'radial-gradient(rgba(20, 85, 30, 0.35) 22%, transparent 24%)',
        borderRadius: '50%',
      };
    });

    const checkedKingSq = findCheckedKingSquare(viewedGame);
    if (checkedKingSq) {
      styles[checkedKingSq] = {
        ...styles[checkedKingSq],
        background: 'radial-gradient(ellipse at center, rgb(255, 0, 0) 0%, rgb(231, 0, 0) 25%, rgba(169, 0, 0, 0) 89%, rgba(158, 0, 0, 0) 100%)',
      };
    }

    Object.keys(rightClickedSquares).forEach((sq) => {
      styles[sq] = rightClickedSquares[sq];
    });

    return styles;
  }, [currentSnapshot, moveFrom, isViewingHistory, legalMovesForSelected, viewedGame, rightClickedSquares]);

  const aiCaptured = playerColor === 'w' ? material.whiteCaptured : material.blackCaptured;
  const aiAdvantage = playerColor === 'w' ? material.blackAdvantage : material.whiteAdvantage;
  const userCaptured = playerColor === 'w' ? material.blackCaptured : material.whiteCaptured;
  const userAdvantage = playerColor === 'w' ? material.whiteAdvantage : material.blackAdvantage;

  return (
    <div className="lichess-container">
      {/* COLONNA SCACCHIERA */}
      <div className="board-column">
        <div className="player-info">
          <div className="player-title">
            <span className="status-dot" data-thinking={kasparovThinking} />
            <span>{kasparovThinking ? '🧠 Kasparov sta riflettendo...' : '🤖 TorchFish (Kasparov-Net)'}</span>
          </div>
          <div className="captured-pieces">
            {aiCaptured.map((p, i) => <span key={`ai-${i}`}>{p}</span>)}
            {aiAdvantage > 0 && <span className="advantage">+{aiAdvantage}</span>}
          </div>
        </div>

        <div className="board-container" onContextMenu={(e) => e.preventDefault()}>
          <Chessboard
            position={currentSnapshot.fen}
            onPieceDrop={onDrop}
            onSquareClick={onSquareClick}
            onSquareRightClick={onSquareRightClick}
            onPieceDragBegin={onPieceDragBegin}
            boardOrientation={playerColor === 'w' ? 'white' : 'black'}
            customSquareStyles={customSquareStyles}
            showBoardNotation={true}
            arePiecesDraggable={!kasparovThinking && !isViewingHistory && !gameOverStatus}
            areArrowsAllowed={true}
            customArrowColor="rgba(255, 170, 0, 0.85)"
            {...(aiCandidateArrows ? { customArrows: aiCandidateArrows } : {})}
          />

          {/* SELETTORE GRAFICO PROMOZIONE PEDONE */}
          {pendingPromotion && (
            <div className="promotion-backdrop" onClick={() => setPendingPromotion(null)}>
              <div className="promotion-modal" onClick={(e) => e.stopPropagation()}>
                <div className="promotion-title">Scegli il pezzo per la promozione</div>
                <div className="promotion-options">
                  {PROMOTION_PIECES.map((type) => (
                    <button
                      key={type}
                      className="promotion-btn"
                      onClick={() => commitUserMove(pendingPromotion.from, pendingPromotion.to, type)}
                    >
                      {PIECE_SYMBOLS[pendingPromotion.color][type]}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* OVERLAY FINE PARTITA */}
          {gameOverStatus && (
            <div className="gameover-banner">
              <div className="gameover-title">{gameOverStatus.title}</div>
              <div className="gameover-subtitle">{gameOverStatus.subtitle}</div>
              <button className="ctrl-btn primary" onClick={() => startNewGame(playerColor)}>
                Gioca Ancora
              </button>
            </div>
          )}
        </div>

        <div className="player-info">
          <div className="player-title">
            <span>👤 Tu ({playerColor === 'w' ? 'Bianco' : 'Nero'})</span>
            {isViewingHistory && (
              <span className="history-badge" onClick={() => setViewIndex(snapshots.length - 1)}>
                ⏪ Storico (Torna live)
              </span>
            )}
          </div>
          <div className="captured-pieces">
            {userCaptured.map((p, i) => <span key={`usr-${i}`}>{p}</span>)}
            {userAdvantage > 0 && <span className="advantage">+{userAdvantage}</span>}
          </div>
        </div>
      </div>

      {/* COLONNA LATERALE: TELEMETRIA, STORICO E CONTROLLI */}
      <div className="sidebar-column">
        <div className="telemetry-panel">
          <div className="telemetry-header">
            <span>⚡ Policy Network (4544 UCI)</span>
            <div className="telemetry-tools">
              <button
                className={`tool-toggle ${showAIArrows ? 'active' : ''}`}
                onClick={() => setShowAIArrows((prev) => !prev)}
                title="Mostra/Nascondi frecce delle mosse candidate di Kasparov"
              >
                🎯 Frecce IA
              </button>
              <button
                className={`tool-toggle ${soundEnabled ? 'active' : ''}`}
                onClick={() => setSoundEnabled((prev) => !prev)}
                title={soundEnabled ? 'Disattiva audio' : 'Attiva audio Lichess'}
              >
                {soundEnabled ? '🔊' : '🔇'}
              </button>
              {telemetry.confidence !== null && (
                <span className="confidence-pill">{telemetry.confidence}%</span>
              )}
            </div>
          </div>
          {telemetry.topMoves.length > 0 ? (
            <div className="candidate-list">
              {telemetry.topMoves.map((cand, idx) => (
                <div key={cand.uci || idx} className="candidate-row">
                  <span className="candidate-san">{formatMoveToFigurine(cand.san)}</span>
                  <div className="candidate-bar-track">
                    <div
                      className="candidate-bar-fill"
                      style={{ width: `${Math.min(100, cand.prob)}%` }}
                    />
                  </div>
                  <span className="candidate-prob">{cand.prob}%</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="telemetry-empty">
              Trascina col tasto destro sulla scacchiera per disegnare frecce tattiche
            </div>
          )}
        </div>

        <div className="nav-Controls">
          <button
            className="nav-btn"
            disabled={viewIndex === 0}
            onClick={() => setViewIndex(0)}
            title="Inizio partita (Freccia Su)"
          >
            |&lt;
          </button>
          <button
            className="nav-btn"
            disabled={viewIndex === 0}
            onClick={() => setViewIndex((i) => Math.max(0, i - 1))}
            title="Mossa precedente (Freccia Sinistra)"
          >
            &lt;
          </button>
          <button
            className="nav-btn"
            disabled={viewIndex === snapshots.length - 1}
            onClick={() => {
              const nextIdx = Math.min(snapshots.length - 1, viewIndex + 1);
              if (nextIdx !== viewIndex && snapshots[nextIdx]) {
                playMoveSound(snapshots[nextIdx], new Chess(snapshots[nextIdx].fen));
              }
              setViewIndex(nextIdx);
            }}
            title="Mossa successiva (Freccia Destra)"
          >
            &gt;
          </button>
          <button
            className="nav-btn"
            disabled={viewIndex === snapshots.length - 1}
            onClick={() => setViewIndex(snapshots.length - 1)}
            title="Ultima mossa (Freccia Giù)"
          >
            &gt;|
          </button>
        </div>

        <div className="move-list" ref={moveListRef}>
          {movePairs.map((pair) => (
            <div key={pair.moveNumber} className="move-row">
              <span className="move-number">{pair.moveNumber}</span>
              <span
                className={`move-cell ${viewIndex === pair.white.index ? 'active-move' : ''}`}
                onClick={() => setViewIndex(pair.white.index)}
              >
                {formatMoveToFigurine(pair.white.san)}
              </span>
              <span
                className={`move-cell ${pair.black && viewIndex === pair.black.index ? 'active-move' : ''}`}
                onClick={() => pair.black && setViewIndex(pair.black.index)}
              >
                {pair.black ? formatMoveToFigurine(pair.black.san) : ''}
              </span>
            </div>
          ))}
        </div>

        <div className="action-bar">
          <div className="action-row">
            <button className="ctrl-btn primary" onClick={() => startNewGame(playerColor)}>
              Nuova Partita
            </button>
            <button className="ctrl-btn" onClick={handleFlipColor} title="Inverti colore e scacchiera">
              🔄 Inverti
            </button>
            <button
              className="ctrl-btn danger"
              onClick={handleResign}
              disabled={Boolean(gameOverStatus) || snapshots.length < 2}
            >
              🏳️ Abbandona
            </button>
          </div>
          <div className="action-row export-row">
            <button className="export-btn" onClick={() => copyToClipboard('FEN')}>
              {copiedBadge === 'FEN' ? '✓ FEN Copiato!' : 'Copia FEN'}
            </button>
            <button className="export-btn" onClick={() => copyToClipboard('PGN')}>
              {copiedBadge === 'PGN' ? '✓ PGN Copiato!' : 'Copia PGN'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}