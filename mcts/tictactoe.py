#!/usr/bin/env python3
"""
Tic-Tac-Toe with pluggable AI (random AI provided).
Player marks: 'X' (player 1) and 'O' (player 2 / AI).
Board indices (for user input): 0..8 (row-major)
You can replace RandomAI.select_move with your MCTS selector later.
"""

import random
from typing import List, Optional, Tuple

# --- Game logic -------------------------------------------------------------

class TicTacToe:
    def __init__(self, board: Optional[List[Optional[str]]] = None):
        # board is a list of 9 elements: 'X', 'O', or None
        self.board = board[:] if board is not None else [None] * 9

    def clone(self) -> "TicTacToe":
        return TicTacToe(self.board)

    def legal_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v is None]

    def apply_move(self, move: int, player: str) -> None:
        if not (0 <= move < 9):
            raise ValueError("move must be 0..8")
        if self.board[move] is not None:
            raise ValueError("square already taken")
        self.board[move] = player

    def is_winner(self, player: str) -> bool:
        b = self.board
        wins = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # cols
            (0,4,8), (2,4,6)            # diags
        ]
        return any(all(b[i] == player for i in triple) for triple in wins)

    def is_draw(self) -> bool:
        return all(v is not None for v in self.board) and not (self.is_winner('X') or self.is_winner('O'))

    def game_over(self) -> bool:
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def result(self) -> Optional[str]:
        """Return 'X' or 'O' if that player won, 'draw' if draw, or None if ongoing."""
        if self.is_winner('X'):
            return 'X'
        if self.is_winner('O'):
            return 'O'
        if self.is_draw():
            return 'draw'
        return None

    def render(self) -> None:
        def cell(v):
            return v if v is not None else ' '
        rows = [
            f" {cell(self.board[0])} | {cell(self.board[1])} | {cell(self.board[2])} ",
            f" {cell(self.board[3])} | {cell(self.board[4])} | {cell(self.board[5])} ",
            f" {cell(self.board[6])} | {cell(self.board[7])} | {cell(self.board[8])} ",
        ]
        sep = "---+---+---"
        print("\n".join([rows[0], sep, rows[1], sep, rows[2]]))

    def get_next_player(self):
        x_count = sum(1 for v in self.board if v == 'X')
        o_count = sum(1 for v in self.board if v == 'O')
        return 'X' if x_count == o_count else 'O'

# --- AI interface & RandomAI ------------------------------------------------

class BaseAI:
    """
    Base AI class. Implement select_move(board, player) to return an int 0..8.
    board: TicTacToe object (can be cloned and simulated).
    player: 'X' or 'O' -- the AI's symbol.
    """
    def select_move(self, board: TicTacToe, player: str) -> int:
        raise NotImplementedError

class RandomAI(BaseAI):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, board: TicTacToe, player: str) -> int:
        moves = board.legal_moves()
        if not moves:
            raise ValueError("No legal moves to select")
        return self.rng.choice(moves)

# --- Game loop helpers -----------------------------------------------------

def prompt_move_from_user(board: TicTacToe) -> int:
    legal = board.legal_moves()
    while True:
        try:
            text = input(f"Enter your move (0..8). Available: {legal}: ").strip()
            move = int(text)
            if move in legal:
                return move
            print("That square is not available. Try again.")
        except ValueError:
            print("Please enter an integer between 0 and 8.")

def play_game(ai_x: Optional[BaseAI], ai_o: Optional[BaseAI], human_first = True) -> str:
    """
    ai_x: AI controlling X or None if human plays X.
    ai_o: AI controlling O or None if human plays O.
    Returns final result: 'X', 'O', or 'draw'
    """
    game = TicTacToe()
    current = 'X'
    while not game.game_over():
        print("\nCurrent board:")
        game.render()

        ai = ai_x if current == 'X' else ai_o
        if ai is None:
            move = prompt_move_from_user(game)
        else:
            move = ai.select_move(game.clone(), current)
            print(f"[AI-{current}] selects {move}")

        game.apply_move(move, current)
        current = 'O' if current == 'X' else 'X'

    print("\nFinal board:")
    game.render()
    res = game.result()
    if res == 'draw':
        print("Game finished: draw.")
    else:
        print(f"Game finished: {res} wins!")
    return res

# --- Main entrypoint -------------------------------------------------------

def main():
    from mcts import MCTSAI
    print("Tic-Tac-Toe")
    print("1) Human vs AI (you are X)")
    print("2) Human vs Human")
    print("3) AI vs AI (MCTS vs Random)")
    print("4) AI vs AI (MCTS vs MCTS)")
    mode = input("Choose mode (1/2/3/4): ").strip()
    if mode == '1':
        ai_x = None
        ai_o = MCTSAI(time_limit=2.0)
        play_game(ai_x, ai_o)
    elif mode == '2':
        play_game(None, None)
    elif mode == '3':
        ai_x = MCTSAI(time_limit=1.0)
        ai_o = RandomAI(seed=42)
        play_game(ai_x, ai_o)
    elif mode == '4':
        ai_x = MCTSAI(time_limit=1.0)
        ai_o = MCTSAI(time_limit=1.0)
        play_game(ai_x, ai_o)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
