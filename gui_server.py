"""Local web GUI: play the trained models, or watch model-vs-model /
model-vs-engine games. Reuses the same loading and move-selection code as
the CLI eval scripts (net_from_state_dict, select_moves_with_lookahead,
select_moves_from_policy) so play here matches evaluate_vs_*.py exactly.
"""
import os
import threading
import uuid
from typing import Dict, Optional

import chess
import chess.engine
import torch
from flask import Flask, jsonify, render_template, request

if torch.version.hip is not None:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from helper import board_to_tensor, index_to_move, legal_moves_mask
from lookahead import select_moves_with_lookahead
from models import net_from_state_dict

MODELS_DIRS = ["models", "models_big"]
DEFAULT_ENGINE_PATH = "./stockfish/stockfish"

app = Flask(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_net_cache: Dict[str, torch.nn.Module] = {}
_net_lock = threading.Lock()

_games: Dict[str, dict] = {}
_games_lock = threading.Lock()


def list_models():
    found = []
    for d in MODELS_DIRS:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".pt"):
                found.append(os.path.join(d, name))
    return found


def get_net(path: str):
    with _net_lock:
        net = _net_cache.get(path)
        if net is None:
            state = torch.load(path, map_location=_device)
            net = net_from_state_dict(state, _device)
            net.eval()
            _net_cache[path] = net
        return net


def _raw_move(net, board: chess.Board, temperature: float) -> chess.Move:
    state = board_to_tensor(board).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits, _ = net(state)
        masked = logits[0].masked_fill(~legal_moves_mask(board).to(_device), -1e9)
        if temperature > 0:
            dist = torch.distributions.Categorical(logits=masked / temperature)
            idx = int(dist.sample().item())
        else:
            idx = int(masked.argmax().item())
    move = index_to_move(idx, board)
    return move if move is not None and move in board.legal_moves else next(iter(board.legal_moves))


def _search_move(net, board: chess.Board, temperature: float, k: int, alpha: float, value_weight: float) -> chess.Move:
    with torch.no_grad():
        idxs, *_ = select_moves_with_lookahead(
            net, [board], _device, top_k=k, alpha=alpha, temperature=temperature, value_weight=value_weight,
        )
    move = index_to_move(int(idxs[0].item()), board)
    return move if move is not None and move in board.legal_moves else next(iter(board.legal_moves))


class ModelPlayer:
    """One side of a game controlled by a checkpoint, raw or search-wrapped."""

    def __init__(self, path: str, raw: bool, temperature: float, k: int, alpha: float, value_weight: float):
        self.path = path
        self.net = get_net(path)
        self.raw = raw
        self.temperature = temperature
        self.k = k
        self.alpha = alpha
        self.value_weight = value_weight

    def pick_move(self, board: chess.Board) -> chess.Move:
        if self.raw:
            return _raw_move(self.net, board, self.temperature)
        return _search_move(self.net, board, self.temperature, self.k, self.alpha, self.value_weight)

    def label(self) -> str:
        mode = "raw" if self.raw else f"search k={self.k} a={self.alpha} vw={self.value_weight}"
        return f"{os.path.basename(self.path)} [{mode}]"


class EnginePlayer:
    def __init__(self, engine_path: str, skill: int, move_time: float):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        try:
            self.engine.configure({"Skill Level": int(skill)})
        except Exception:
            pass
        self.skill = skill
        self.move_time = move_time

    def pick_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=max(self.move_time, 0.01)))
        return result.move

    def label(self) -> str:
        return f"Stockfish (skill={self.skill}, {self.move_time}s)"

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def _result_info(board: chess.Board) -> dict:
    if not board.is_game_over():
        return {"game_over": False, "result": None}
    return {"game_over": True, "result": board.result()}


def _game_state(game: dict) -> dict:
    board: chess.Board = game["board"]
    info = _result_info(board)
    return {
        "game_id": game["id"],
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "last_move": game.get("last_move"),
        "white_label": game["white_label"],
        "black_label": game["black_label"],
        "human_color": game.get("human_color"),
        "move_number": board.fullmove_number,
        "san_history": game.get("san_history", []),
        **info,
    }


def _mover_for(game: dict, color: bool):
    return game["white"] if color == chess.WHITE else game["black"]


def _apply_move(game: dict, move: chess.Move):
    board: chess.Board = game["board"]
    san = board.san(move)
    board.push(move)
    game["last_move"] = move.uci()
    game.setdefault("san_history", []).append(san)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models()})


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    body = request.get_json(force=True)
    mode = body["mode"]  # "human_vs_model" | "model_vs_model" | "model_vs_engine"
    game_id = uuid.uuid4().hex[:12]
    board = chess.Board()

    def build_model_player(cfg):
        return ModelPlayer(
            path=cfg["path"],
            raw=cfg.get("raw", True),
            temperature=float(cfg.get("temperature", 1.0)),
            k=int(cfg.get("k", 4)),
            alpha=float(cfg.get("alpha", 1.0)),
            value_weight=float(cfg.get("value_weight", 2.0)),
        )

    game = {"id": game_id, "board": board, "mode": mode, "san_history": []}

    if mode == "human_vs_model":
        human_color = body.get("human_color", "white")
        model_player = build_model_player(body["model"])
        game["human_color"] = human_color
        if human_color == "white":
            game["white"] = "human"
            game["black"] = model_player
            game["white_label"] = "Human"
            game["black_label"] = model_player.label()
        else:
            game["white"] = model_player
            game["black"] = "human"
            game["white_label"] = model_player.label()
            game["black_label"] = "Human"
    elif mode == "model_vs_model":
        model_a = build_model_player(body["model_a"])
        model_b = build_model_player(body["model_b"])
        game["white"] = model_a
        game["black"] = model_b
        game["white_label"] = model_a.label()
        game["black_label"] = model_b.label()
    elif mode == "model_vs_engine":
        model_player = build_model_player(body["model"])
        engine_cfg = body.get("engine", {})
        engine_player = EnginePlayer(
            engine_path=engine_cfg.get("path", DEFAULT_ENGINE_PATH),
            skill=int(engine_cfg.get("skill", 0)),
            move_time=float(engine_cfg.get("move_time", 0.1)),
        )
        model_color = body.get("model_color", "white")
        if model_color == "white":
            game["white"] = model_player
            game["black"] = engine_player
            game["white_label"] = model_player.label()
            game["black_label"] = "Stockfish"
        else:
            game["white"] = engine_player
            game["black"] = model_player
            game["white_label"] = "Stockfish"
            game["black_label"] = model_player.label()
    else:
        return jsonify({"error": f"unknown mode {mode!r}"}), 400

    with _games_lock:
        _games[game_id] = game
    return jsonify(_game_state(game))


@app.route("/api/state")
def api_state():
    game_id = request.args.get("game_id")
    game = _games.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game_id"}), 404
    return jsonify(_game_state(game))


@app.route("/api/human_move", methods=["POST"])
def api_human_move():
    body = request.get_json(force=True)
    game = _games.get(body["game_id"])
    if game is None:
        return jsonify({"error": "unknown game_id"}), 404
    board: chess.Board = game["board"]
    if board.is_game_over():
        return jsonify(_game_state(game))

    uci = body["uci"]
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify({"error": f"bad uci {uci!r}"}), 400
    if move not in board.legal_moves:
        # Drag-and-drop UIs send bare from/to; auto-promote to queen if that's
        # what makes an otherwise-illegal pawn move legal.
        promo_move = chess.Move.from_uci(uci + "q")
        if promo_move in board.legal_moves:
            move = promo_move
        else:
            return jsonify({"error": "illegal move"}), 400
    _apply_move(game, move)

    # Auto-play the opponent's reply if it's a model/engine, not another human.
    if not board.is_game_over():
        mover = _mover_for(game, board.turn)
        if mover != "human":
            reply = mover.pick_move(board)
            _apply_move(game, reply)

    return jsonify(_game_state(game))


@app.route("/api/step", methods=["POST"])
def api_step():
    """Advance one ply for model-vs-model / model-vs-engine games."""
    body = request.get_json(force=True)
    game = _games.get(body["game_id"])
    if game is None:
        return jsonify({"error": "unknown game_id"}), 404
    board: chess.Board = game["board"]
    if board.is_game_over():
        return jsonify(_game_state(game))
    mover = _mover_for(game, board.turn)
    if mover == "human":
        return jsonify({"error": "waiting on human move"}), 400
    move = mover.pick_move(board)
    _apply_move(game, move)
    return jsonify(_game_state(game))


@app.route("/api/end_game", methods=["POST"])
def api_end_game():
    body = request.get_json(force=True)
    with _games_lock:
        game = _games.pop(body.get("game_id"), None)
    if game is not None:
        for side in (game.get("white"), game.get("black")):
            if isinstance(side, EnginePlayer):
                side.close()
    return jsonify({"ok": True})


@app.route("/api/legal_moves")
def api_legal_moves():
    game_id = request.args.get("game_id")
    game = _games.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game_id"}), 404
    board: chess.Board = game["board"]
    return jsonify({"moves": [m.uci() for m in board.legal_moves]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
