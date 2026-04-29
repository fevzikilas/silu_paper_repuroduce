from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from environments.base import Environment


Action = Tuple[int, int]


@dataclass(frozen=True)
class Afterstate:
    action: Action
    board: np.ndarray
    features: np.ndarray
    lines_cleared: int
    holes: int
    reward: float


class SZTetris(Environment):
    def __init__(self, seed: int | None = None, encoding: str = "threshold460"):
        self.width = 10
        self.height = 20
        self.height_bins = 21
        self.diff_min = -10
        self.diff_max = 10
        self.diff_bins = self.diff_max - self.diff_min + 1
        self.hole_bins = 61
        self.encoding = encoding
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.pieces = {
            "S": (
                np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
                np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
            ),
            "Z": (
                np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
                np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int8),
            ),
        }
        self.current_piece_type = "S"
        self.total_lines_cleared = 0
        self._legal_afterstates_cache: List[Afterstate] | None = None
        self._afterstate_by_action_cache: dict[Action, Afterstate] | None = None

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_piece_type = self._sample_piece()
        self.total_lines_cleared = 0
        self._invalidate_afterstate_cache()
        return self._board_observation()

    def step(self, action: Action):
        afterstate = self.simulate_action(action)
        if afterstate is None:
            return self._board_observation(), -1.0, True, {"score": self.total_lines_cleared, "lines_cleared": 0}

        self.board = afterstate.board.copy()
        self.total_lines_cleared += afterstate.lines_cleared
        self.current_piece_type = self._sample_piece()
        self._invalidate_afterstate_cache()
        next_afterstates = self.get_legal_afterstates()
        done = len(next_afterstates) == 0
        info = {
            "score": self.total_lines_cleared,
            "lines_cleared": afterstate.lines_cleared,
            "holes": afterstate.holes,
        }
        return self._board_observation(), afterstate.reward, done, info

    def get_legal_afterstates(self) -> List[Afterstate]:
        if self._legal_afterstates_cache is not None:
            return self._legal_afterstates_cache

        afterstates: List[Afterstate] = []
        for rotation, piece in enumerate(self.pieces[self.current_piece_type]):
            max_x = self.width - piece.shape[1]
            for x in range(max_x + 1):
                landing_y = self._find_landing_y(self.board, piece, x)
                if landing_y is None:
                    continue
                placed_board = self._place_piece(self.board, piece, x, landing_y)
                cleared_board, lines_cleared = self._clear_lines(placed_board)
                holes = self._count_holes(cleared_board)
                reward = float(np.exp(-holes / 33.0))
                features = self._encode_features(cleared_board)
                afterstates.append(
                    Afterstate(
                        action=(rotation, x),
                        board=cleared_board,
                        features=features,
                        lines_cleared=lines_cleared,
                        holes=holes,
                        reward=reward,
                    )
                )
        self._legal_afterstates_cache = afterstates
        self._afterstate_by_action_cache = {afterstate.action: afterstate for afterstate in afterstates}
        return afterstates

    def simulate_action(self, action: Action) -> Afterstate | None:
        self.get_legal_afterstates()
        if self._afterstate_by_action_cache is None:
            return None
        return self._afterstate_by_action_cache.get(action)

    def _invalidate_afterstate_cache(self) -> None:
        self._legal_afterstates_cache = None
        self._afterstate_by_action_cache = None

    def _sample_piece(self) -> str:
        return str(self.rng.choice(["S", "Z"]))

    def _board_observation(self) -> np.ndarray:
        return self.board.astype(np.float32, copy=True)

    def _find_landing_y(self, board: np.ndarray, piece: np.ndarray, x: int) -> int | None:
        y = 0
        if not self._valid_position(board, piece, x, y):
            return None
        while self._valid_position(board, piece, x, y + 1):
            y += 1
        return y

    def _valid_position(self, board: np.ndarray, piece: np.ndarray, x: int, y: int) -> bool:
        for row in range(piece.shape[0]):
            for col in range(piece.shape[1]):
                if piece[row, col] == 0:
                    continue
                board_x = x + col
                board_y = y + row
                if board_x < 0 or board_x >= self.width or board_y < 0 or board_y >= self.height:
                    return False
                if board[board_y, board_x] != 0:
                    return False
        return True

    def _place_piece(self, board: np.ndarray, piece: np.ndarray, x: int, y: int) -> np.ndarray:
        next_board = board.copy()
        for row in range(piece.shape[0]):
            for col in range(piece.shape[1]):
                if piece[row, col] != 0:
                    next_board[y + row, x + col] = 1
        return next_board

    def _clear_lines(self, board: np.ndarray) -> Tuple[np.ndarray, int]:
        full_rows = np.all(board == 1, axis=1)
        lines_cleared = int(np.sum(full_rows))
        if lines_cleared == 0:
            return board, 0
        remaining = board[~full_rows]
        padding = np.zeros((lines_cleared, self.width), dtype=np.int8)
        return np.vstack((padding, remaining)), lines_cleared

    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        heights = np.zeros(self.width, dtype=np.int32)
        for col in range(self.width):
            filled_rows = np.flatnonzero(board[:, col])
            heights[col] = 0 if filled_rows.size == 0 else self.height - int(filled_rows[0])
        return heights

    def _count_holes(self, board: np.ndarray) -> int:
        holes = 0
        for col in range(self.width):
            seen_block = False
            for row in range(self.height):
                if board[row, col] == 1:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes

    def _encode_features(self, board: np.ndarray) -> np.ndarray:
        heights = self._column_heights(board)
        diffs = np.diff(heights)
        holes = min(self._count_holes(board), self.hole_bins - 1)

        if self.encoding == "threshold460":
            return self._encode_threshold460(heights, diffs, holes)
        if self.encoding == "onehot460":
            return self._encode_onehot460(heights, diffs, holes)
        if self.encoding == "ordinal460":
            return self._encode_ordinal460(heights, diffs, holes)
        raise ValueError(f"Unknown encoding: {self.encoding}")

    def _encode_threshold460(self, heights: np.ndarray, diffs: np.ndarray, holes: int) -> np.ndarray:
        features = np.zeros(460, dtype=np.float32)
        offset = 0

        for height in heights:
            clipped_height = int(np.clip(height, 0, 20))
            if clipped_height > 0:
                features[offset : offset + clipped_height] = 1.0
            offset += 20

        for diff in diffs:
            clipped_diff = int(np.clip(diff, self.diff_min, self.diff_max))
            if clipped_diff < 0:
                features[offset : offset + abs(clipped_diff)] = 1.0
            elif clipped_diff > 0:
                features[offset + 10 : offset + 10 + clipped_diff] = 1.0
            offset += 20

        hole_threshold_bins = 80
        capped_holes = min(holes, hole_threshold_bins)
        if capped_holes > 0:
            features[offset : offset + capped_holes] = 1.0
        return features

    def _encode_onehot460(self, heights: np.ndarray, diffs: np.ndarray, holes: int) -> np.ndarray:
        features = np.zeros(460, dtype=np.float32)
        offset = 0

        for height in heights:
            clipped_height = int(np.clip(height, 0, self.height_bins - 1))
            features[offset + clipped_height] = 1.0
            offset += self.height_bins

        for diff in diffs:
            clipped_diff = int(np.clip(diff, self.diff_min, self.diff_max))
            features[offset + (clipped_diff - self.diff_min)] = 1.0
            offset += self.diff_bins

        features[offset + holes] = 1.0
        return features

    def _encode_ordinal460(self, heights: np.ndarray, diffs: np.ndarray, holes: int) -> np.ndarray:
        features = np.zeros(460, dtype=np.float32)
        offset = 0

        for height in heights:
            clipped_height = int(np.clip(height, 0, self.height_bins - 1))
            features[offset + clipped_height] = 1.0
            offset += self.height_bins

        for diff in diffs:
            clipped_diff = int(np.clip(diff, self.diff_min, self.diff_max))
            features[offset + (clipped_diff - self.diff_min)] = 1.0
            offset += self.diff_bins

        if holes > 0:
            features[offset : offset + holes] = 1.0
        return features