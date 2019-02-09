import tensorflow as tf
import numpy as np
import random
import game
from tensorflow.keras import layers

# global vars
verbose = False

# hyperparams
learning_rate = 0.01
decay_rate = 0
momentum = 0
discount_factor = 0.99

training_batch_size = 100
batches_to_run = 10000

class PolicyModel:
    def __init__(self):
        x = tf.placeholder(tf.float32, shape=(None, 42))
        actions = tf.placeholder(tf.float32, shape=(None, 7))
        rewards = tf.placeholder(tf.float32, shape=(None, 1))

        hidden1 = tf.layers.dense(x, 
            units=200, 
            activation=tf.nn.relu,
            bias_initializer=tf.initializers.glorot_normal,
            kernel_initializer=tf.initializers.glorot_normal)
        hidden2 = tf.layers.dense(hidden1, 
            units=200, 
            activation=tf.nn.relu,
            bias_initializer=tf.initializers.glorot_normal,
            kernel_initializer=tf.initializers.glorot_normal)
        logit_layer = tf.layers.Dense(units=7, 
            kernel_initializer=tf.initializers.glorot_normal)
        # logits = tf.layers.dense(hidden, 
        #     units=7,
        #     kernel_initializer=tf.initializers.random_normal)
        logits = logit_layer(hidden2)
        out = tf.nn.softmax(logits)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=actions, logits=logits)
        loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy))

        optimizer = tf.train.RMSPropOptimizer(learning_rate, 
            decay=decay_rate, momentum=momentum).minimize(loss)

        self.x = x
        self.actions = actions
        self.rewards = rewards
        self.out = out
        self.optimizer = optimizer
        self.logit_layer = logit_layer

    def act(self, board):
        probs = sess.run(self.out, feed_dict={self.x: board.get_features()})
        move = sample_from_probabilities(probs)
        return move

    def act_optimally(self, board):
        probs = sess.run(self.out, feed_dict={self.x: board.get_features()})
        vprint(probs)
        move = np.argmax(probs)
        return move

    def optimize(self, sess, x_stack, actions_stack, rewards_stack):
        sess.run(self.optimizer, feed_dict={
            self.x: x_stack,
            self.actions: actions_stack,
            self.rewards: rewards_stack})

class RandomModel:
    def act(self, board):
        return random.randint(0, 6)
    def act_optimally(self, board):
        return random.randint(0, 6)
    def optimize(self, sess, x_stack, actions_stack, rewards_stack):
        pass

def vprint(arg):
    if verbose:
        print(arg)

def sample_from_probabilities(probs):
    return np.random.choice(np.arange(probs.shape[1]), p=probs[0, :])

def run_training_batch(sess, p1_model, p2_model):
    p1_state = {
        'board_states': [],
        'actions': [],
        'rewards': [],
        'last_game_boundary': 0
    }

    p2_state = {
        'board_states': [],
        'actions': [],
        'rewards': [],
        'last_game_boundary': 0
    }

    # run training_batch_size games
    for i in range(training_batch_size):
        vprint("Running training episode: %d / %d" % (i, training_batch_size))
        run_training_episode(sess, p1_model, p1_state, p2_model, p2_state)

    # run policy gradient update
    update_model(sess, p1_model, p1_state)
    update_model(sess, p2_model, p2_state)

def run_training_episode(sess, p1_model, p1_state, p2_model, p2_state):
    # initialize board
    board = game.Board()

    # run game until someone wins or the game draws
    while board.get_winner() == -1:
        vprint("Round %d, Player %d" % (board.round, board.turn))
        if board.turn == 1:
            run_move(sess, board, p1_model, p1_state)
        elif board.turn == 2:
            run_move(sess, board, p2_model, p2_state)
        vprint(np.flip(board.get_board(), 0))
    vprint(board.get_winner())

    # fill in the discounted rewards
    compute_discounted_rewards(board.get_reward(1), p1_state)
    compute_discounted_rewards(board.get_reward(2), p2_state)

def run_move(sess, board, model, state):
    cur_board = np.copy(board.get_features())
    # sample move and act
    move = model.act(board)
    board.play(move)
    # add action vector
    action = np.zeros((1, 7))
    action[0, move] = 1

    state['board_states'].append(cur_board)
    state['actions'].append(action)
    state['rewards'].append(0)

def run_evaluation_episode(sess, p1_model, p2_model):
    # initialize board
    board = game.Board()

    # run game until someone wins or the game draws
    while board.get_winner() == -1:
        vprint("Round %d, Player %d" % (board.round, board.turn))
        if board.turn == 1:
            run_eval_move(sess, board, p1_model)
        elif board.turn == 2:
            run_eval_move(sess, board, p2_model)
        vprint(np.flip(board.get_board(), 0))

    vprint(board.get_winner())
    return board.get_winner()

def run_eval_move(sess, board, model):
    move = model.act_optimally(board)
    board.play(move)

def compute_discounted_rewards(reward, state):
    game_boundary = state['last_game_boundary']
    rewards_len = len(state['rewards'])
    state['rewards'][-1] = reward
    running_sum = state['rewards'][rewards_len - 1]
    for i in reversed(range(game_boundary, rewards_len - 1)):
        running_sum = running_sum * discount_factor
        state['rewards'][i] = running_sum
    state['last_game_boundary'] = len(state['rewards'])

def update_model(sess, model, state):
    # create vertical stacks for each of the state elements
    x_stack = np.vstack(state['board_states'])
    actions_stack = np.vstack(state['actions'])
    rewards_stack = np.vstack(state['rewards'])

    # run optimizer
    model.optimize(sess, x_stack, actions_stack, rewards_stack)

if __name__ == '__main__':
    # verbose = True

    # initialize models
    p1_model = PolicyModel()
    p2_model = RandomModel()

    # initialize session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    # saver.restore(sess, "./checkpoints/model.ckpt")

    for i in range(batches_to_run):
        print("Running training batches: %d / %d" % (i, batches_to_run))
        run_training_batch(sess, p1_model, p2_model)

        if i % 10 == 0:
            save_path = saver.save(sess, "./checkpoints/model.ckpt")


    # now run a game
    verbose = True
    wins = 0
    for i in range(1000):
        winner = run_evaluation_episode(sess, p1_model, p2_model)
        if winner == 1:
            wins += 1

    print(wins)