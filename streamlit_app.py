import numpy as np
import random
import streamlit as st

# Initialize game state
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((5, 5), dtype=int)
    st.session_state.current_player = 1
    st.session_state.winner = None
    st.session_state.q_agent = None

# Function to check for a winner
def check_winner(board):
    for i in range(5):
        if np.all(board[i, :] == board[i, 0]) and board[i, 0] != 0:
            return board[i, 0]
        if np.all(board[:, i] == board[0, i]) and board[0, i] != 0:
            return board[0, i]
    if np.all(board.diagonal() == board[0, 0]) and board[0, 0] != 0:
        return board[0, 0]
    if np.all(np.fliplr(board).diagonal() == board[0, 4]) and board[0, 4] != 0:
        return board[0, 4]
    if not np.any(board == 0):
        return 0  # Draw
    return None

# Function to reset the game
def reset_game():
    st.session_state.board = np.zeros((5, 5), dtype=int)
    st.session_state.current_player = 1
    st.session_state.winner = None

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
    
    def get_state(self, board):
        return tuple(board.flatten())
    
    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.q_table.get((state, action), 0) for action in available_actions]
        max_q_value = max(q_values)
        return random.choice([action for action, q in zip(available_actions, q_values) if q == max_q_value])
    
    def learn(self, state, action, reward, next_state, next_available_actions):
        old_q_value = self.q_table.get((state, action), 0)
        future_q_values = [self.q_table.get((next_state, next_action), 0) for next_action in next_available_actions]
        max_future_q_value = max(future_q_values, default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
        self.q_table[(state, action)] = new_q_value

# Initialize the Q-learning agent
if st.session_state.q_agent is None:
    st.session_state.q_agent = QLearningAgent()

# Function to make a move
def make_move(row, col):
    if st.session_state.board[row, col] == 0 and st.session_state.winner is None:
        st.session_state.board[row, col] = st.session_state.current_player
        st.session_state.winner = check_winner(st.session_state.board)
        if st.session_state.winner is None:
            st.session_state.current_player = 3 - st.session_state.current_player

# Function to train the Q-learning agent
def train_agent():
    for episode in range(1000):  # Training for 1000 games
        board = np.zeros((5, 5), dtype=int)
        agent = st.session_state.q_agent
        current_player = 1
        winner = None

        while winner is None:
            state = agent.get_state(board)
            available_actions = [(r, c) for r in range(5) for c in range(5) if board[r, c] == 0]
            if current_player == 1:  # Agent's turn
                action = agent.choose_action(state, available_actions)
            else:  # Random move for the opponent
                action = random.choice(available_actions)
            
            board[action[0], action[1]] = current_player
            next_state = agent.get_state(board)
            winner = check_winner(board)

            if winner == 1:
                reward = 1
                agent.learn(state, action, reward, next_state, available_actions)
                break
            elif winner == 2:
                reward = -1
                agent.learn(state, action, reward, next_state, available_actions)
                break
            elif winner == 0:
                reward = 0
                agent.learn(state, action, reward, next_state, available_actions)
                break
            
            if current_player == 1:
                agent.learn(state, action, 0, next_state, available_actions)
            current_player = 3 - current_player

# Train the agent before the game starts
train_agent()

# Display the game board
for row in range(5):
    cols = st.columns(5)
    for col in range(5):
        if st.session_state.board[row, col] == 0:
            cols[col].button(" ", key=f"{row}-{col}", on_click=make_move, args=(row, col))
        else:
            cols[col].button("X" if st.session_state.board[row, col] == 1 else "O", disabled=True)

# Display game status
if st.session_state.winner is not None:
    if st.session_state.winner == 0:
        st.write("It's a draw!")
    else:
        st.write(f"Player {st.session_state.winner} wins!")
    if st.button("Restart Game"):
        reset_game()
