import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from q_learning_agent import Agent 

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
    agent = Agent(lr=0.001, gamma = 0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_dec=0.999995, n_actions=env.action_space.n, n_states=env.observation_space.n)

    scores = []
    win_pct_list = []
    n_games = 200000

    for i in tqdm(range(n_games)):
        done = False
        observation,_ = env.reset()
        score = 0 
        while not done:
            action = agent.choose_action(state=observation)
            observation_,reward,terminated,truncated,info = env.step(action=action)
            done = terminated or truncated
            agent.learn(state=observation,action=action,reward=reward,state_=observation_)
            score += reward
            observation = observation_

        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print(f"episode: {i} win pct: {win_pct} epsilon: {agent.epsilon}")
                # tqdm.set_postfix(ordered_dict={'episode': i, 'win pct': win_pct, 'epsilon':agent.epsilon})
    plt.plot(win_pct_list)
    plt.show()
