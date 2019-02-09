import numpy as np

class Board:
    def __init__(self):
        # 6x7 grid of {0, 1, 2} for empty, p1, p2 respectively.
        # (0, 0) is defined as the bottom left corner of the board.
        self.board = np.zeros((6,7), dtype=int)
        # player 1 or 2
        self.turn = 1
        self.round = 1
        self.winner = -1

    # Play a piece in the given column, change the board state,
    # and advance the player number. If the move is illegal,
    # throw an exception.
    def play(self, col):
        row = 0
        while row < 6 and self.board[row, col] != 0:
            row += 1
        if row == 6:
            # the player attempted to play in a full column,
            # so they lose the game.
            self.winner = 1 if self.turn == 2 else 2
            return

        self.board[row, col] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
            self.round += 1

        self.update_winner()


    # return the board as a 1D feature array
    def get_features(self):
        return np.reshape(self.board, (1, 42))

    def get_board(self):
        return self.board

    # updates self.winner to 1 if player 1 has won the game, 2 if player 2 has won the game,
    # 0 if the game is a draw, or -1 if the game is not over yet.
    def update_winner(self):
        # vertical victory
        for i in range(0, 3):
            for j in range(0, 7):
                if self.board[i, j] == 0:
                    continue
                if np.all(self.board[i:i+4, j] == self.board[i, j]):
                    self.winner = self.board[i, j]
                    return

        # horizontal victory
        for i in range(0, 6):
            for j in range(0, 4):
                if self.board[i, j] == 0:
                    continue
                if np.all(self.board[i, j:j+4] == self.board[i, j]):
                    self.winner = self.board[i, j]
                    return

        # / victory
        for i in range(0, 3):
            for j in range(0, 4):
                if self.board[i, j] == 0:
                    continue
                if np.all(np.array([self.board[i+a, j+a] for a in range(4)]) == self.board[i, j]):
                    self.winner = self.board[i, j]
                    return

        # \ victory
        for i in range(0, 3):
            for j in range(3, 4):
                if self.board[i, j] == 0:
                    continue
                if np.all(np.array([self.board[i+a, j-a] for a in range(4)]) == self.board[i, j]):
                    self.winner = self.board[i, j]
                    return

        # draw
        if np.all(self.board != 0):
            self.winner = 0
            return

        self.winner = -1

    def get_winner(self):
        return self.winner

    def get_reward(self, player):
        if self.winner == -1:
            return 0
        elif self.winner == 0:
            return 0.5
        elif self.winner == player:
            return 1
        else:
            return -1

    def __str__(self):
        return "Round %d : Turn %d\n%s" % (self.round, self.turn, np.flip(self.board, 0))