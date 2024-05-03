
import numpy as np
from dqn_model import DQN
from environment import StockTradingEnv

def train_dqn(episodes, batch_size):
    # Initialize the trading environment with your chosen stock symbol and API key
    env = StockTradingEnv('AAPL', 'WRGH4XCNUZA425TB')  
    state_size = env.state_size
    action_size = 3  # Define action space size (e.g., Buy, Sell, Hold)
    
    # Initialize DQN model
    dqn = DQN(state_size, action_size)
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        while True:
            action = dqn.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {episode + 1}/{episodes}, Total reward: {total_reward}, Portfolio value: {info['current_portfolio_value']}")
                break

            if len(dqn.memory) > batch_size:
                dqn.replay(batch_size)

        # Save the model every 50 episodes
        if episode % 50 == 0:
            dqn.save(f"dqn_model_{episode}.h5")

if __name__ == '__main__':
    episodes = 1000  # Number of episodes for training
    batch_size = 32  # Number of experiences to sample from memory
    train_dqn(episodes, batch_size)
