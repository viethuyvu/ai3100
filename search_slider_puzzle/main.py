import sys

GOAL = "12345678x"

def generate_connected_states(state):
    connected_states = []
    state_idx = state.index("x")
    row, col = divmod(state_idx, 3)

    return row, col

def main():
    board = "".join(line.strip() for line in sys.stdin)
    print(generate_connected_states(board))

if __name__ == "__main__":
    main()