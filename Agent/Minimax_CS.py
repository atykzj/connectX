import numpy as np
import random

def cell_swarm_minimax_agent(obs, config):
    N_STEPS = 5
    
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid
    
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        alpha, beta = -1e4, 1e6
        score = minimax(next_grid, nsteps-1, False, mark, alpha, beta, config)
        return score
    
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow
    
    def is_terminal_node(grid, config):
        if list(grid[0, :]).count(0) == 0:
            return True
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False
    
    def minimax(node, depth, maximizingPlayer, mark, alpha, beta, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config), None
        if maximizingPlayer:
            value = -np.Inf
            move = None
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                child_value, _ = minimax(child, depth-1, False, mark, alpha, beta, config)
                if child_value > value:
                    value = child_value
                    move = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, move
        else:
            value = np.Inf
            move = None
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                child_value, _ = minimax(child, depth-1, True, mark, alpha, beta, config)
                if child_value < value:
                    value = child_value
                    move = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
                return value, move
    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
        return score

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

    def get_cell_swarm(num_agents, max_depth):
        swarm = []
        for i in range(num_agents):
            depth = random.randint(1, max_depth)
            agent = {'id': i, 'depth': depth}
            swarm.append(agent)
        return swarm
    def swarm_agent(obs, config):
        global swarm
        if obs['step'] == 0:
            # Initialize the swarm
            swarm = get_cell_swarm(num_agents=10, max_depth=4)
        else:
            # Update the swarm with the result of the last move
            last_move = obs['last_action']
            for agent in swarm:
                if agent['move'] == last_move:
                    agent['score'] += obs['reward']
            
        # Evaluate the swarm on each available move
        grid = np.asarray(obs['board']).reshape(config.rows, config.columns)
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        scores = {c: 0 for c in valid_moves}
        for agent in swarm:
            depth = agent['depth']
            mark = obs['mark']
            for col in valid_moves:
                score = score_move(grid, col, mark, config, depth)
                scores[col] += score
        
        # Choose the move with the highest average score
        best_moves = [move for move in scores.keys() if scores[move] == max(scores.values())]
        return random.choice(best_moves)

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

        
    # # define swarm's and opponent's marks
    # swarm_mark = obs.mark
    # opp_mark = 2 if swarm_mark == 1 else 1
    # # define swarm's center
    # swarm_center_horizontal = conf.columns // 2
    # swarm_center_vertical = conf.rows // 2
    
    # # define swarm as two dimensional array of cells
    # swarm = []
    # for column in range(conf.columns):
    #     swarm.append([])
    #     for row in range(conf.rows):
    #         cell = {
    #                     "x": column,
    #                     "y": row,
    #                     "mark": obs.board[conf.columns * row + column],
    #                     "swarm_patterns": {},
    #                     "opp_patterns": {},
    #                     "distance_to_center": abs(row - swarm_center_vertical) + abs(column - swarm_center_horizontal),
    #                     "points": []
    #                 }
    #         swarm[column].append(cell)
    
    # best_cell = None
    # # start searching for best_cell from swarm center
    # x = swarm_center_horizontal
    # # shift to right or left from swarm center
    # shift = 0
    
    # # searching for best_cell
    # while x >= 0 and x < conf.columns:
    #     # find first empty cell starting from bottom of the column
    #     y = conf.rows - 1
    #     while y >= 0 and swarm[x][y]["mark"] != 0:
    #         y -= 1
    #     # if column is not full
    #     if y >= 0:
    #         # current cell evaluates its own qualities
    #         current_cell = evaluate_cell(swarm[x][y])
    #         # current cell compares itself against best cell
    #         best_cell = choose_best_cell(best_cell, current_cell)
                        
    #     # shift x to right or left from swarm center
    #     if shift >= 0:
    #         shift += 1
    #     shift *= -1
    #     x = swarm_center_horizontal + shift

    # # return index of the best cell column
    # return best_cell["x"]

