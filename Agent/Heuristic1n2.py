
################################
# Imports and helper functions #
################################

import numpy as np
import random

def Heuristic1n2(obs, config):
    """
    In this Minimax-AB algorithm, for EVERY valid moves available
    we calculate up to N_STEPS search depth summing get_heuristic score
    which is the sum of each of its recursive nodes. At every step, we keep max
    of the score (alpha) which is the best move, and prune all moves where beta<alpha,
    where alpha is the player, beta is the opponent.
    Depending if you are the player or opponent, the score and alpha/beta is opposite
     because we go through each child node, which may be either of the players' positions.
    """
    # Trying with Alpha–beta pruning implementation. N_STEPS = 5 leads to timeout if submited without Alpha–beta pruning. 
    N_STEPS = 3
    
    # scoring matrix
    score_mat = [
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3]
    ]

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
        alpha, beta = -float('inf'), float('inf')
        score = minimax(next_grid, nsteps-1, False, mark, alpha, beta, config)
        return score

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
    
    # # Minimax implementation with Alpha-Beta pruning
    def minimax(node, depth, maximizingPlayer, mark, alpha, beta, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config), None
        # Agent's turn
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
        playerTurn = 0
        positionScoreMatrix = 0
        for row in range(config.rows):
            for col in range(config.columns):
                if grid[row][col] == 1:
                    playerTurn += 1
                    positionScoreMatrix += score_mat[row][col]
                elif grid[row][col] == 2:
                    playerTurn -= 1
                    positionScoreMatrix -= score_mat[row][col]
        playerTurn += 1
        score = 0
        
        # horizontal
        for row in range(config.rows):
            score += get_score(grid[row,:], mark, playerTurn)
            score -= get_score(grid[row,:], mark%2+1, playerTurn)
        # vertical
        for col in range(config.columns):
            score += get_score(grid[:,col], mark, playerTurn)
            score -= get_score(grid[:,col], mark%2+1, playerTurn)
            
        # positive diagonal
        for row in range(config.rows-3):
            diagonal = [grid[row,0]]
            i = row + 1
            j = 1
            while i < config.rows and j < config.columns:
                diagonal.append(grid[i,j])
                i += 1
                j += 1
            score += get_score(diagonal, mark, playerTurn)
            score -= get_score(diagonal, mark%2+1, playerTurn)
        for col in range(1, config.columns-3):
            diagonal = [grid[0,col]]
            i = 1
            j = col + 1
            while i < config.rows and j < config.columns:
                diagonal.append(grid[i,j])
                i += 1
                j += 1
            score += get_score(diagonal, mark, playerTurn)
            score -= get_score(diagonal, mark%2+1, playerTurn)
            
        # negative diagonal
        for row in range(3, config.rows):
            diagonal = [grid[row,0]]
            i = row - 1
            j = 1
            while i >= 0 and j < config.columns:
                diagonal.append(grid[i,j])
                i -= 1
                j += 1
            score += get_score(diagonal, mark, playerTurn)
            score -= get_score(diagonal, mark%2+1, playerTurn)
        for col in range(1, config.columns-3):
            diagonal = [grid[config.rows-1,col]]
            i = config.rows-2
            j = col + 1
            while i >= 0 and j < config.columns:
                diagonal.append(grid[i,j])
                i -= 1
                j += 1
            score += get_score(diagonal, mark, playerTurn)
            score -= get_score(diagonal, mark%2+1, playerTurn)
            
        # Add current position matrix score
        return score + 138 + positionScoreMatrix
    
    def last_index(l,start):
        # returns the last consecutive index that has the same value as l[start]
        end = start+1
        while end<len(l) and l[end] == l[start]:
            end += 1
        return (end-1)
    
    def get_score(l, piece, playerTurn):
        start = 0
        score = 0
        while start < len(l):
            if l[start] == piece:
                end = last_index(l,start)
                length = end-start+1
                if length >= 4:
                    return float('inf')
                elif length == 3:
                    if start>0 and end<len(l)-1:
                        if l[start-1] == 0 and l[end+1] == 0:
                            return 100000000
                        if l[start-1] == 0 or l[end+1] == 0:
                            if playerTurn == piece:
                                return 100000000
                            score += 100000
                    if start == 0:
                        if l[end+1] == 0:
                            score += 100000
                    if end == len(l)-1:
                        if l[start] == 0:
                            score += 100000
                elif length == 2:
                    if start>1 and l[start-2] == piece:
                        score += 100000
                    elif end<len(l)-2 and l[end+2] == 0:
                        score += 100000
                    else:
                        free_spaces = 0
                        if start>0 and l[start-1] == 0:
                            free_spaces += 1
                            if start>1 and l[start-2] == 0:
                                free_spaces += 1
                        if end<len(l)-1 and l[end+1] == 0:
                            free_spaces += 1
                            if end<len(l)-2 and l[end+2] == 0:
                                free_spaces += 1
                        score += min(free_spaces-1,0) * 1000
                elif length == 1:
                    score += 10
                start = end
            start += 1
        return score        

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