import numpy as np
import random

class CellSwarm:
    def __init__(self, rows, cols, mark):
        self.rows = rows
        self.cols = cols
        self.mark = mark
        self.centroids = self.get_centroids()

    def get_centroids(self):
        centroids = []
        for i in range(self.cols):
            centroids.append([self.rows-1, i])
        return centroids

    def distance_to_swarm(self, grid, col):
        distances = []
        for i in range(self.cols):
            if i != col:
                row = grid[:, i].argmin() - 1
                if row < 0:
                    row = 0
                centroid = self.centroids[i]
                distance = np.sqrt((centroid[0] - row)**2 + (centroid[1] - i)**2)
                distances.append(distance)
        return min(distances)

    def distance_to_nearest_cell(self, grid, col):
        row = grid[:, col].argmin() - 1
        if row < 0:
            row = 0
        centroid = self.centroids[col]
        distance = np.sqrt((centroid[0] - row)**2 + (centroid[1] - col)**2)
        return distance

    def distance_to_farthest_cell(self, grid, col):
        row = grid[:, col].argmax()
        centroid = self.centroids[col]
        distance = np.sqrt((centroid[0] - row)**2 + (centroid[1] - col)**2)
        return distance

    def num_possible_wins(self, grid, col, mark, config):
        row = grid[:, col].argmin() - 1
        if row < 0:
            return 0
        wins = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = row, col
                discs = 0
                while r >= 0 and r < config.rows and c >= 0 and c < config.columns and grid[r][c] == mark:
                    discs += 1
                    r += i
                    c += j
                if discs > 0 and r >= 0 and r < config.rows and c >= 0 and c < config.columns and grid[r][c] == 0:
                    wins += 1
        return wins

def Minimax_AB_CellSwarm(obs, config):
    # Helper function for cell swarm call
    def cell_swarm_heuristic(grid, col, mark, config):
        cswarm = CellSwarm(config.rows, config.columns, mark)
        next_grid = drop_piece(grid, col, mark, config)
        feature_weights = [0.1, 1, 2, 5]
        features = [cswarm.distance_to_swarm(next_grid, col), cswarm.distance_to_nearest_cell(next_grid, col),
                    cswarm.distance_to_farthest_cell(next_grid, col), cswarm.num_possible_wins(next_grid, col, mark, config)]
        score = np.dot(features, feature_weights)
        return score

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid
    
    # Uses minimax to calculate value of dropping piece in selected column
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        alpha, beta = -1e4, 1e6
        score, move = minimax(next_grid, nsteps-1, False, mark, alpha, beta, config)
        score += cell_swarm_heuristic(next_grid, col, mark, config)
        return score, move

    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows
    
    # Helper function for minimax: calculate heuristic value for terminal node
    def evaluate_window(window, mark, config):
        score = 0
        opp_mark = 3 - mark
        if window.count(mark) == config.inarow:
            score += 100
        elif window.count(mark) == config.inarow-1 and window.count(0) == 1:
            score += 10
        elif window.count(mark) == config.inarow-2 and window.count(0) == 2:
            score += 5
        if window.count(opp_mark) == config.inarow-1 and window.count(0) == 1:
            score -= 7
        return score
    
    def check_winning_move(board, mark):
        # Check horizontal locations
        for row in range(config.rows):
            for col in range(config.columns-3):
                if board[row][col] == mark and board[row][col+1] == mark and board[row][col+2] == mark and board[row][col+3] == mark:
                    return True

        # Check vertical locations
        for row in range(config.rows-3):
            for col in range(config.columns):
                if board[row][col] == mark and board[row+1][col] == mark and board[row+2][col] == mark and board[row+3][col] == mark:
                    return True

        # Check positively sloped diaganols
        for row in range(config.rows-3):
            for col in range(config.columns-3):
                if board[row][col] == mark and board[row+1][col+1] == mark and board[row+2][col+2] == mark and board[row+3][col+3] == mark:
                    return True

        # Check negatively sloped diaganols
        for row in range(3, config.rows):
            for col in range(config.columns-3):
                if board[row][col] == mark and board[row-1][col+1] == mark and board[row-2][col+2] == mark and board[row-3][col+3] == mark:
                    return True

        return False


    def get_next_open_row(board, col):
        for r in range(config.rows-1, -1, -1):
            if board[r][col] == 0:
                return r
        return None


    def switch_mark(mark):
        if mark == 1:
            return 2
        else:
            return 1

    # Uses minimax with alpha-beta pruning to calculate value of dropping piece in selected column
    def minimax(grid, depth, maximizing_player, mark, alpha, beta, config):
        # base case: reached maximum depth or game is over
        is_terminal = is_terminal_node(grid, config)
        if depth == 0 or is_terminal:
            if is_terminal:
                if check_winning_move(grid, mark, config):
                    return (None, 100000000000000)
                elif check_winning_move(grid, 3 - mark, config):
                    return (None, -10000000000000)
                else:  # Game is over, no more valid moves
                    return (None, 0)
            else:  # Depth is zero
                last_move = 0
                for col in range(config.columns):
                    if grid[0][col] == 1:
                        last_move = col
                return None, cell_swarm_heuristic(grid, last_move, mark, config)

        # recursive case: call minimax on all possible moves and choose the best move
        valid_moves = [col for col in range(config.columns) if grid[0][col] == 0]
        if maximizing_player:
            value = -1e6
            best_move = random.choice(valid_moves)
            for col in valid_moves:
                row = get_next_open_row(grid, col, config)
                next_grid = drop_piece(grid, col, mark, config)
                score, _ = minimax(next_grid, depth-1, False, mark, alpha, beta, config)
                score += cell_swarm_heuristic(next_grid, col, mark, config)
                if score > value:
                    value = score
                    best_move = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:  # minimizing player
            value = 1e6
            best_move = random.choice(valid_moves)
            for col in valid_moves:
                row = get_next_open_row(grid, col, config)
                next_grid = drop_piece(grid, col, switch_mark(mark), config)
                score, _ = minimax(next_grid, depth-1, True, mark, alpha, beta, config)
                score += cell_swarm_heuristic(next_grid, col, mark, config)
                if score < value:
                    value = score
                    best_move = col
                beta = min(beta, value)
                if alpha >= beta:
                    break

        return best_move, value

    # RUN ALGORITHM
    # Get mark
    mark = obs.mark
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Use minimax to find best move
    nsteps = 5
    # best_score = -1e6
    # best_col = random.choice([col for col in range(config.columns) if grid[0][col] == 0])
    # for col in range(config.columns):
    #     if grid[0][col] == 0:
    #         score, _ = score_move(grid, col, mark, config, nsteps)
    #         if score > best_score:
    #             best_score = score
    #             best_col = col
    # move = best_col

    # return move
        # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, mark, config, nsteps) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

