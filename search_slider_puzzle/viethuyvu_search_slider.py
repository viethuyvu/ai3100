import sys
from collections import deque

GOAL = "12345678x"

def generate_connected_states(state):
    connected_states = []
    state_idx = state.index("x")
    row, col = divmod(state_idx, 3)

    moves = {
        "d": (-1, 0),
        "u": (1, 0),
        "r": (0, -1),
        "l": (0, 1)
    }

    for dir, move in moves.items():
        new_row = row + move[0]
        new_col = col + move[1]
        if 0<= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row*3+new_col
            state_list = list(state)
            state_list[state_idx], state_list[new_idx] = state_list[new_idx], state_list[state_idx]
            connected_states.append(("".join(state_list), dir))

    return connected_states

def bfs(start):
    if start == GOAL:
        return ""
    q = deque([(start, "")])
    visited = set()
    visited.add(start)

    while q:
        state, path = q.popleft()
        for new_state, dir in generate_connected_states(state):
            if new_state not in visited:
                if new_state == GOAL:
                    return path + dir
                visited.add(new_state)
                q.append((new_state, path + dir))

    return "unsolvable"

def main():
    board = "".join(line.strip() for line in sys.stdin)
    print(bfs(board))

if __name__ == "__main__":
    main()