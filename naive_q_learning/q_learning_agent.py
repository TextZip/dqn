import numpy as np

class Agent():
    def __init__(self,lr, gamma, n_actions, n_states, epsilon_start, epsilon_end, epsilon_dec):

        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state,action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state,a)] for a in range(self.n_actions)])
            action = np.random.choice(np.flatnonzero(actions == actions.max()))
        return action 

    def decrement_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end \
            else self.epsilon_end

    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        a_max = np.random.choice(np.flatnonzero(actions == actions.max()))

        self.Q[(state,action)] += self.lr*(reward + self.gamma*self.Q[(state_, a_max)] - self.Q[(state,action)])
        self.decrement_epsilon()





