import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import json
import numpy as np

with open('config.json') as f:
    data = f.read()
config = json.loads(data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class Environment:
    """A epidemic infection simulation model.
    Attributes
    ----------
        is_terminal: 
        _start: 
                
        S: number of susceptible
        E: number of exposed
        E_move: number of exposed from moving
        Q: number of quarantine
        Q_move: number of quarantine from moving
        I: number of infectious
        R: number of recovered
        
        gamma_mask: the factor of masks to affect transmission rate
        gamma_recover: the factor of mask trade to affect recover rate
        gamma_detect: the factor of mask trade to affect COVID detection
        gamma_move: the factor of moving to affect population of moving
        gamma_shut: the factor of shutdown to affect infectious degree
    Methods
    -------
        init_state: Prepare the environment for the starting state of a trial.
        step: Make a state transition.
        obeserved_state: To transform a global state to an obsevable state for agent.
        update_gamma_mask: To update the value of gamma_mask
        update_gamma_move: To update the value of gamma_move
        start(property): Return the starting starting state of a trial.
    """
    def __init__(self):
        self._start = torch.zeros(8).to(device)
        self.is_terminal = False
    
    def init_state(self):
        
        """ We need to define this function to initialize the environment.
        Returns:
            The starting state of the environment

        """
        self.S, self.E, self.E_move, self.Q, self.Q_move, self.I, self.R = config['S0'], config['E0'], config['E0_move'], config['Q0'], config['Q0_move'], config['I0'], config['R0']
        self._start[0] = config['N_mask0']
        self._start[1] = config['N_transparency0']
        self._start[2] = self.Q + self.Q_move
        self._start[3] = 0
        self._start[4] = 0
        self._start[5] = config['N_value0']
        self._start[6] = 0
        self._start[7] = 0
        
        self.gamma_mask = config['gamma_mask0']
        self.gamma_recover = config['gamma_recover0']
        self.gamma_detect = config['gamma_detect0']
        self.gamma_move = config['gamma_move0']
        self.gamma_shut = config['gamma_shut0']
        self.is_terminal = False

        return self.start

   
    def step(self, s, a, t):
        """ We need to define this function as the transition function.(SEIR....)

        Args:
            s: current state
                s[0]: number of masks
                s[1]: number of transparency
                s[2]: number of total quarantine
                s[3]: 1: shutdown / 0: not shutdown
                s[4]: 1: close moving / 0: open moving
                s[5]: number of value       
                s[6]: recording transparency bias
                s[7]: recording transparency rate

            a: action taken by agent in current state
                0: mask trade
                1: transparency to 0.5 (current transparency should be smaller than 0.5)
                2: transparency to 1.0 (current transparency should be smaller than 1.0)
                3: dereasing transparency (transparency should not be 0)
                4: switch shutdown mode
                5: switch moving mode
                6: no action
            t: time(per day)

        Returns:
            next_sate: The next state (aka  s' ) given by s and a .
            reward of (s, a, s')

        """
        
        """ update state """
        s = s.clone()
        
        # recording tranparency bias and rate
        if t == config['early_threshold']:
            s[6] = (2 * s[1] + 1) / (s[1] + 1)
            s[7] = s[1] / (s[1] + 1)
        
        s[0] = (t * (config['MAX_mask'] - 0.1) / config['early_threshold'] + 0.1) if t <= config['early_threshold'] else config['MAX_mask']
        s[5] = max(0, s[5] - config['shut_rate'] * t) if s[3] else min(config['N_value0'], s[5] + 0.7 * config['shut_rate'] * t)
        if a == 0:
            s[0] = (s[0] * config['N_total'] - config['N_donate']) / config['N_total']
            # AIT
            if np.random.uniform() < 0.01:
                s[5] += config['Up_mask']
            
            # Medical technology improvement
            if np.random.uniform() < 0.01:
                self.gamma_recover += config['Up_recover']
                self.gamma_detect += config['Up_detect']

            # update gamma_mask
                self.update_gamma_mask(s)
        
        # modify transparency to 0.5
        elif a == 1:
            s[1] = 0.5   
            self.update_gamma_mask(s)
            self.update_gamma_move(s, t)
        
        # modify transparency to 1.0
        elif a == 2:
            s[1] = 1.0   
            self.update_gamma_mask(s)
            self.update_gamma_move(s, t)
        
        # decrease transparency 
        elif a == 3:
            s[1] -= 0.1   
            self.update_gamma_mask(s)
            self.update_gamma_move(s, t)

        # switch shutdown mode
        elif a == 4:
            s[3] = int(not s[3])
       
        # switch moving mode
        elif a == 5:
            s[4] = int(not s[4])
            s[5] = s[5] - config['Up_move'] if s[4] else s[5] + config['Up_move']
        
        """ update SEIR """
        beta = self.gamma_mask * config['beta0']

        SI = int(config['rI0'] * beta * self.I * self.S / config['N_total'])
        SE = int(self.gamma_shut * config['rE0'] * beta * self.E * self.S / config['N_total'])
        SE_move = int(beta * self.S * self.gamma_move * config['P_move0'] * (1 - 0.85 * s[4]))
        
        EI = int(config['alpha_ei0'] * self.E)
        EI_move = int(config['alpha_ei_move0'] * self.E_move)
        EQ = int(self.E - np.random.binomial(self.E, max(0, 1 - self.gamma_detect * config['alpha_eq0']))) if (self.Q < config['MAX_Q']) else 0
        if (self.Q + EQ) > config['MAX_Q']:
            EQ  = config['MAX_Q'] - self.Q
        EQ_move = int(self.E_move - np.random.binomial(self.E_move, max(0, 1 - self.gamma_detect * config['alpha_eq_move0']))) if (self.Q_move < config['MAX_Q_move']) else 0
        if self.Q_move + EQ_move > config['MAX_Q_move']:
            EQ_move = config['MAX_Q_move'] - self.Q_move
        
        QI = int(config['alpha_qi0'] * self.Q)
        QI_move = int(config['alpha_qi_move0'] * self.Q_move)
        
        IR = int(self.gamma_recover * config['alpha_ir0'] * self.I)

        self.S = self.S - SI - SE - SE_move
        self.E = self.E + SI + SE - EI - EQ
        self.E_move = self.E_move + SE_move - EI_move - EQ_move
        self.Q = self.Q + EQ - QI 
        self.Q_move = self.Q_move + EQ_move - QI_move 
        self.I = self.I + EI + EI_move + QI + QI_move - IR
        self.R = self.R + IR
        s[2] = self.Q + self.Q_move

        """ update reward """
        reward = 5566

        return s, reward, self.is_terminal

    def obeserved_state(self, state):
        """ To transform the global state to the obsevable state for agent.

        Args:
            state : global state

        Returns:
            observed_state : obsevable state for agent

        """
        return state[:5].clone()
    
    def update_gamma_mask(self, s):
        self.gamma_mask = min((1 - 0.8 * s[1]) * config['MAX_mask'] / s[0], 1)

    def update_gamma_move(self, s, t):
        return (s[1] + 1) if t <= config['early_threshold'] else (s[6] - s[7] * s[1])
    
    @property
    def start(self):
        return self._start


class Agent(nn.Module):
    """ The decision-making policy network for the simulation.

    Attributes
    ----------
        init_legal_actions:

        legal_actions:

        log_probs:

        rewards:

        action_embeddings:

        net:

    Methods
    -------
        init_agent: reset the history fo the log probability and rewards

        forward: the inference of the policy network

        select_actions: return the action sampled from the policy nwetwork

    """
    def __init__(self, dim_input, dim_output, max_actions, init_legal_actions):
        """ Initialize the parameters of the policynwtwork

        Args
        ----
            dim_input:
            dim_output:
            max_actions:
            init_legal_actions:

        """

        super(Agent, self).__init__()

        self.init_legal_actions = init_legal_actions

        self.legal_actions = self.init_legal_actions.copy()

        self.log_probs = []

        self.rewards = []

        self.action_embeddings = nn.Embedding(max_actions, dim_output)

        self.net = nn.Sequential(
            nn.Linear(dim_input, 8),
            nn.Dropout(0.1),
            nn.Linear(8, dim_output)
        )

    def init_agent(self):

        self.legal_actions = self.init_legal_actions.copy()

        self.log_probs = []

        self.rewards = []

    def forward(self, state):

        state_vector = self.net(state)

        actions = torch.tensor(self.legal_actions, device=device)

        actions_vectors = self.action_embeddings(actions)

        scores = torch.matmul(actions_vectors, state_vector.unsqueeze(1)).squeeze()

        return F.softmax(scores, dim=0)

    def select_actions(self, scores):

        distribution = Categorical(scores)

        action = distribution.sample()

        self.log_probs.append(distribution.log_prob(action))
        
        return action.item()


class Simulatoin:
    """ The interaction between environment(epidemic infection model) and agent(policy network)

    Attributes
    ----------
        agent
        enviroment
        gamma
        optimizer

    Methods
    -------
        policy_gradient_update:

        episodes:
    """

    def __init__(self, agent: Agent, environment: Environment, optimizer=None):

        self.agent = agent

        self.environment = environment

        self.gamma = 0.9

        self.optimizer = optimizer
    
    def policy_gradient_update(self):

        policy_loss = 0
        accumulated_rewards = []

        # v_t = r_t + gammar*v_{t+1}
        # v_T = 0
        # agent.rewards : [r_0, r_1, r_2, ........ r_T]

        v = 0   # V_T = 0

        for r in self.agent.rewards[::-1]:   # r_T, r_{T-1}, .....r_0

            v = r + self.gamma * v

            accumulated_rewards.insert(0, v)

        accumulated_rewards = torch.tensor(accumulated_rewards, device=device)

        accumulated_rewards = (accumulated_rewards - accumulated_rewards.mean()) / (accumulated_rewards.std() + 1e-9)

        for log_prob, r in zip(self.agent.log_probs, accumulated_rewards):

            policy_loss += (-log_prob * r)

        self.optimizer.zero_grad()
        
        policy_loss.backward()

        self.optimizer.step()

    def episodes(self, max_episodes=3, max_steps=10):

        for episode in range(max_episodes):

            state = self.environment.init_state()

            state_observed = self.environment.obeserved_state(state)

            reward_episode = 0

            for t in range(max_steps):

                actions_probs = self.agent.forward(state_observed)

                action = self.agent.select_actions(actions_probs)

                state, reward, is_terminal = self.environment.step(state, action, t=t)

                state_observed = self.environment.obeserved_state(state)

                self.agent.rewards.append(reward)

                reward_episode += reward

                if is_terminal:

                    break

            self.policy_gradient_update()

            self.agent.init_agent()


def main():

    init_legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    args_agent = {
        "dim_input": 5, 
        "dim_output": 3,
        "max_actions": 10,
        "init_legal_actions": init_legal_actions
    }

    agent = Agent(**args_agent).to(device)

    env = Environment()

    game = Simulatoin(agent, env, optim.Adam(agent.parameters(), lr=1e-3))

    game.episodes()


if __name__ == "__main__":

    main()
