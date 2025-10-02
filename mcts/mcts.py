import math
import random
from typing import List, Optional
import time

from tictactoe import TicTacToe, BaseAI

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
        if not self.children:
            raise ValueError("No children to select from")
        
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
        if not self.untried_moves:
            raise ValueError("No moves left to expand")
        
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        next_player = self.game_state.get_next_player()
        new_game_state = self.game_state.clone()
        new_game_state.apply_move(move, next_player)
        child_node = MCTSNode(new_game_state, parent=self, move=move, player_who_moved=next_player)
        self.children.append(child_node)
        return child_node
    
    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits) if self.children else None
    
class MCTSAI:
    def __init__(self, time_limit: float = 1.0, exploration_constant: float = 1.41):
        self.time_limit = time_limit
        print(f"MCTS AI initialized with time limit: {self.time_limit} seconds")
        self.exploration_constant = exploration_constant

    def result_to_reward(self,result, root_player):
        if result == 'draw':
            return 0.5
        elif result == root_player:
            return 1.0
        else:
            return 0.0

    def selection(self, node:MCTSNode):
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(self.exploration_constant)
        return node
    
    def expansion(self, node:MCTSNode):
        if not node.is_terminal() and not node.is_fully_expanded():
            return node.expand()
        return node
    
    def simulation(self, game_state:TicTacToe, root_player):
        sim_state = game_state.clone()
        while not sim_state.game_over():
            moves = sim_state.legal_moves()
            if moves:
                move = random.choice(moves)
                next_player = sim_state.get_next_player()
                sim_state.apply_move(move, next_player)

        return self.result_to_reward(sim_state.result(), root_player)
    
    def backpropagation(self, node:MCTSNode, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def select_move(self, board:TicTacToe, player):
        if board.game_over():
            raise ValueError("Game is already over")
        root = MCTSNode(board.clone())
        start_time = time.time()

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
