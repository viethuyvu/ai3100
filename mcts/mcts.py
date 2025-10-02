import math
import random
from typing import List
import time
import os

from tictactoe import TicTacToe

"""Monte Carlo Tree Search (MCTS) AI for Tic-Tac-Toe
The MTCS Process is shown inside of the select_move function:"""

class MCTSNode:
    def __init__(self, game_state:TicTacToe, parent=None, move=None, player_who_moved=None):
        self.game_state = game_state 
        self.parent = parent 
        self.move = move 
        self.children: List[MCTSNode] = []
        self.player_who_moved = player_who_moved
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = game_state.legal_moves()
    
    def is_terminal (self):
        return self.game_state.game_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_constant: float = 1.41):
        # Select the child with the highest UCB1 score
        
        log_parent_visits = math.log(max(1, self.visits))
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                # UCB1 formula
                exploitation = child.total_reward / child.visits
                exploration = exploration_constant * math.sqrt(log_parent_visits / child.visits)
                ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child
    
    def expand(self):
        # Expand by creating a new child node for one of the untried moves
        if not self.untried_moves:
            return None  # No moves to expand
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        next_player = self.game_state.get_next_player()
        new_game_state = self.game_state.clone()
        new_game_state.apply_move(move, next_player)
        child_node = MCTSNode(new_game_state, parent=self, move=move, player_who_moved=next_player)
        self.children.append(child_node)
        return child_node
    
    def most_visited_child(self):
        # Return the child with the highest visit count
        return max(self.children, key=lambda c: c.visits) if self.children else None
    
class MCTSAI:
    def __init__(self, time_limit: float = 1.0, exploration_constant: float = 1.41):
        self.time_limit = time_limit
        print(f"MCTS AI initialized with time limit: {self.time_limit} seconds")
        self.exploration_constant = exploration_constant

    def result_to_reward(self,result, root_player):
        # Convert game result to reward from the perspective of root_player
        if result == 'draw':
            return 0.5
        elif result == root_player:
            return 1.0
        else:
            return 0.0

    def selection(self, node:MCTSNode):
        # Select until we reach a node that is not fully expanded or is terminal
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(self.exploration_constant)
        return node
    
    def expansion(self, node:MCTSNode):
        # Expand the node if it is not terminal and not fully expanded by calling expand method in MCTSNode
        if not node.is_terminal() and not node.is_fully_expanded():
            return node.expand()
        return node
    
    def simulation(self, game_state:TicTacToe, root_player):
        # Simulate a random playout from the given game state to a terminal state
        sim_state = game_state.clone()
        while not sim_state.game_over():
            moves = sim_state.legal_moves()
            if moves:
                move = random.choice(moves)
                next_player = sim_state.get_next_player()
                sim_state.apply_move(move, next_player)

        return self.result_to_reward(sim_state.result(), root_player)
    
    def backpropagation(self, node:MCTSNode, reward):
        # Backpropagate the result up to the root node
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def select_move(self, board:TicTacToe, player):
        # Main MCTS loop
        root = MCTSNode(board.clone())
        start_time = time.time()
        # Run simulations until time limit is reached

        while time.time() - start_time < self.time_limit:
            # Selection
            node = self.selection(root)

            # Expansion
            if not node.is_terminal():
                node = node.expand()

            # Simulation
            reward = self.simulation(node.game_state, player)

            # Backpropagation
            self.backpropagation(node, reward)

        best_child = root.most_visited_child()
        if best_child is None:
            return random.choice(board.legal_moves())
        
        return best_child.move
