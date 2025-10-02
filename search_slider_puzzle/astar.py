import sys
import heapq

GOAL = "12345678x"

def manhattan_distance(state):
    distance = 0
    for idx, value in enumerate(state):
        if value != 'x':
            target_idx = int(value) - 1
            current_row, current_col = divmod(idx, 3)
            target_row, target_col = divmod(target_idx, 3)
            distance += abs(current_row - target_row) + abs(current_col - target_col)
    return distance

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

def astar(start):
    pd = []
    g = 0
    h = manhattan_distance(start)
    heapq.heappush(pd, (g + h, g, start, ""))
    visited = {start: g}

    while pd:
        f,g,state,path = heapq.heappop(pd)
        if state == GOAL:
            return path
        for new_state, dir in generate_connected_states(state):
            new_g = g + 1
            if new_state not in visited or new_g < visited[new_state]:
                visited[new_state] = new_g
                new_h = manhattan_distance(new_state)
                heapq.heappush(pd, (new_g + new_h, new_g, new_state, path + dir))

    return "unsolvable"

def main():
    board = "".join(line.strip() for line in sys.stdin)
    print(astar(board))

if __name__ == "__main__":
    main()