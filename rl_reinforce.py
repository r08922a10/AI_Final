import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import json
import numpy as np
from constant import *
import collections
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class Environment:

    """A epidemic infection simulation model.
    Attributes
    ----------
        is_terminal: 
        _start: 
        _history: store all records including seir model or other states
                
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

        move_w: the weight to affect the factor of moving(gamma_move)
        move_b: the bias to affect the factor of moving(gamma_move)
    Methods
    -------
        init_state: Prepare the environment for the starting state of a trial.
        step: Make a state transition.
        obeserved_state: To transform a global state to an obsevable state for agent.
        update_gamma_mask: To update gamma_mask
        update_gamma_move: To update gamma_move
        start(property): Return the starting starting state of a trial.
    """
    def __init__(self):

        self._config = self._load_config()

        self._start = torch.zeros(8).to(device)

        self.is_terminal = False

        self._history = collections.defaultdict(list)
    
    def _load_config(self):

        """ Load the environment configuration.
        Returns:
            The dictionary of environment configuration.

        """
        try:

            config = json.load(open('config.json'))

        except:

            raise Exception('Failed to load environment config from \'config.json\'')

        return config
    
    def init_state(self):
        
        """ We need to define this function to initialize the environment.
        Returns:
            The starting state of the environment

        """
        # init SEIR model
        self.S = self._config['S']
        self.E = self._config['E']
        self.E_move = self._config['E_move']
        self.Q = self._config['Q']
        self.Q_move = self._config['Q_move']
        self.I = self._config['I']
        self.R = self._config['R']

        # init state
        self._start[N_MASK] = self._config['N_mask']
        self._start[N_OPEN] = self._config['N_open']
        self._start[N_QUARANTINE] = 0
        self._start[IF_SHUTDOWN] = 0
        self._start[IF_MOVE_CONTROL] = 0
        self._start[N_GOLD] = self._config['N_gold']
        
        # init gamma
        self.gamma_mask = self._config['gamma_mask']
        self.gamma_recover = self._config['gamma_recover']
        self.gamma_detect = self._config['gamma_detect']
        self.gamma_move = self._config['gamma_move']
        self.gamma_shut = self._config['gamma_shut']

        # init terminal state
        self.is_terminal = False

        # init gamma move parameter
        self.move_w = 0
        self.move_b = 0

        self._history = collections.defaultdict(list)

        self._assert_all(if_config=True, if_seir=True)

        return self.start.clone()
   
    def step(self, s, a, t):
        """ We need to define this function as the transition function.(SEIR....)

        Args:
            s: current state
                0: N_MASK, number of masks
                1: N_OPEN, number of openness
                2: N_QUARANTINE, number of total quarantine
                3: IF_SHUTDOWN, 1: shutdown / 0: not shutdown
                4: IF_MOVE_CONTROL, 1: moving control / 0: moveing free
                5: N_GOLDnumber of gold

            a: action taken by agent in current state
                0: TRADE_MASK, Trade masks
                1: SET_OPEN, set openness to 0.5 (current openness should be smaller than 0.5)
                2: SET_OPEN2, set openness to 1.0 (current openness should be smaller than 1.0)
                3: DEC_OPEN, decrease openness (openness should be larger than 0)
                4: SWITCH_SHUTDOWN, switch shutdown mode
                5: SWITCH_MOVE_CONTROL, switch moving mode
                6: NO_ACTION, do nothing
            t: timestep

        Returns:
            next_sate: The next state (aka  s' ) given by s and a .
            reward of (s, a, s')

        """
        
        """ update state """
        self._assert_all(s=s, if_seir=True)

        # record move weight and bias when timestep is at early threshold
        if t == self._config['early_threshold']:

            self.move_b = (2 * s[N_OPEN] + 1) / (s[N_OPEN] + 1)

            self.move_w = s[N_OPEN] / (s[N_OPEN] + 1)

        # update number of mask according to early threshold
        s[N_MASK] += (self._config['MAX_mask'] - 0.1) / self._config['early_threshold']
        
        if s[N_MASK] > self._config['MAX_mask']:
            
            s[N_MASK] = self._config['MAX_mask']

        # update number of gold by determining if it is shutdown
        if s[IF_SHUTDOWN] == 1:

            self.gamma_shut = 0.3

            s[N_GOLD] = max(0, s[N_GOLD] - self._config['shut_rate'] * t)

        else:

            self.gamma_shut = 1

            if s[N_GOLD] < self._config['N_gold']:
                
                s[N_GOLD] = min(self._config['N_gold'], s[N_GOLD] + 0.01 * self._config['shut_rate'] * t)

        # conduct action
        if a == TRADE_MASK:
            
            s[N_MASK] = max((s[N_MASK] * self._config['N_total'] - self._config['N_donate']) / self._config['N_total'], 0)
            
            # it would probably increase gold
            if np.random.uniform() < 0.1:

                s[N_GOLD] += self._config['inc_mask']
            
            # it would probably gain medical technology improvement
            if np.random.uniform() < 0.25:

                self.gamma_recover += self._config['inc_recover']

                self.gamma_recover = min(0.95 / self._config['alpha_ir'], self.gamma_recover)
  
                self.gamma_detect += self._config['inc_detect']

                self.gamma_detect = min(0.95 /  self._config['alpha_eq'], self.gamma_detect)
 
        elif a == SET_OPEN:

            s[N_OPEN] = 0.5
        
        elif a == SET_OPEN2:

            s[N_OPEN] = 1.0
        
        elif a == DEC_OPEN:

            s[N_OPEN] -= 0.1
            
            s[N_OPEN] = max(0, s[N_OPEN]) # should be larger than 0

        elif a == SWITCH_SHUTDOWN:
            
            s[IF_SHUTDOWN] = 1 - s[IF_SHUTDOWN]
       
        elif a == SWITCH_MOVE_CONTROL:

            s[IF_MOVE_CONTROL] = 1 - s[IF_MOVE_CONTROL]

            if s[IF_MOVE_CONTROL] == 1:

                s[N_GOLD] -= self._config['inc_move']

            else:

                s[N_GOLD] += self._config['inc_move']

        self.update_gamma_mask(s)

        self.update_gamma_move(s, t)
        
        """ update SEIR """
        beta = self.gamma_mask * self._config['beta']

        SI = int(self._config['rI'] * beta * self.I * self.S / self._config['N_total'])
        
        SI = min(self.S, SI)
        
        SE = int(self.gamma_shut * self._config['rE'] * beta * self.E * self.S / self._config['N_total'])
        
        SE = min(self.S - SI, SE)

        SE_move = int(beta * self.S * self.gamma_move * self._config['p_move'])

        if s[IF_MOVE_CONTROL] == 1:
            
            SE_move = int(0.15 * SE_move)
            
        SE_move = min(self.S - SI - SE, SE_move)
        
        EI = int(self._config['alpha_ei'] * self.E)

        EI_move = int(self._config['alpha_ei_move'] * self.E_move)

        if s[N_QUARANTINE].item() < self._config['MAX_Q']:
            
            EQ = np.random.binomial(self.E - EI, self.gamma_detect * self._config['alpha_eq'])

            EQ_move = np.random.binomial(self.E_move - EI_move, self.gamma_detect * self._config['alpha_eq_move'])
            
            EQ_move = min(self._config['MAX_Q'] - s[N_QUARANTINE].item(), EQ_move)
     
            EQ = min(self._config['MAX_Q'] - s[N_QUARANTINE].item() - EQ_move, EQ)
               
        else:

            EQ = 0

            EQ_move = 0
                  
        QI = int(self._config['alpha_qi'] * self.Q)

        QI_move = int(self._config['alpha_qi_move'] * self.Q_move)
        
        IR = int(self.gamma_recover * self._config['alpha_ir'] * self.I)

        self.S = self.S - SI - SE - SE_move

        self.E = self.E + SI + SE - EI - EQ

        self.E_move = self.E_move + SE_move - EI_move - EQ_move

        self.Q = self.Q + EQ - QI

        self.Q_move = self.Q_move + EQ_move - QI_move
 
        self.I = self.I + EI + EI_move + QI + QI_move - IR

        self.R = self.R + IR
        
        s[N_QUARANTINE] = self.Q + self.Q_move

        """ update reward """
        reward = s[N_GOLD] - ((self.E + self.E_move) * 0.5 + self.I * 0.5) / self._config['N_total']
        
        if s[N_QUARANTINE] == self._config['MAX_Q']:
           
           reward -= 0.01

        self.update_history(t)

        if self.R / self._config['N_total'] > 0.92:

            self.is_terminal = True

        
        self._assert_all(s=s, if_seir=True, if_variable=True)

        return s, reward, self.is_terminal


    def _assert_all(self, s=None, if_config=False, if_variable=False, if_seir=False, if_all=False):
        """ To assert that all values in environment are reasonable.

        Args:
            s: the state you want to assert values. Note that only required when if_variable is True.
            if_config: if True assert all configs in initialization file are reasonable. 
            if_variable: if True, assert all variables are reasonable when the env updates.
            if_seir: if True, assert all seir components are reasonable when the env updates.
        """
        if if_config or if_all:

            rule1 = lambda x: 0 <= x <= 1

            alpha_list = ['alpha_ei', 'alpha_ir', 'alpha_eq', 'alpha_qi',\
                        'alpha_ei_move', 'alpha_ir_move', 'alpha_eq_move', 'alpha_qi_move']

            other_list = ['beta', 'p_move', 'shut_rate']

            for name in alpha_list + other_list:

                assert rule1(self._config[name])

            rule2 = lambda x: 0 <= x

            inc_list = ['inc_mask', 'inc_recover', 'inc_detect', 'inc_move']

            gamma_list = ['gamma_mask', 'gamma_move', 'gamma_shut', 'gamma_recover', 'gamma_detect']

            constant_list = ['N_mask', 'N_gold', 'N_open', 'MAX_mask', 'MAX_Q', 'early_threshold']

            for name in inc_list + gamma_list + constant_list:

                assert rule2(self._config[name])

            rule3 = lambda x: 1 <= x

            for name in ['rE', 'rI', 'N_total', 'N_donate']:

                assert rule3(self._config[name])
        
        if if_variable or if_all:

            assert s is not None

            assert s[N_MASK] <= self._config['MAX_mask']

            assert 0 <= s[N_OPEN] <= 1

            assert 0 <= s[N_QUARANTINE] <= self._config['MAX_Q']

            assert s[IF_SHUTDOWN] in (0, 1)
           
            assert s[IF_MOVE_CONTROL] in (0, 1)

            # assert 0 <= s[N_GOLD] TODO 
            
            assert 0 <= self.move_w <= 1

            assert 0 <= self.move_b
            
            assert 0 <= self.gamma_shut # TODO

            assert 0 <= self.gamma_recover # TODO

            assert 0 <= self.gamma_detect # TODO

        if if_seir or if_all:

            pop_rule = lambda x: 0 <= x <= self._config['N_total']

            for name in ['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']:

                assert pop_rule(getattr(self, name))

            assert 0 <= self.Q + self.Q_move < self._config['MAX_Q']


    def obeserved_state(self, state):
        """ To transform the global state to the obsevable state for agent.

        Args:
            state : global state

        Returns:
            observed_state : obsevable state for agent

        """
        return state[:5].clone()
    
    def update_gamma_mask(self, s):
        """ To update gamma_mask via openness and max of masks constant.

        Args:
            s : global state

        """
        self.gamma_mask = (1 - 0.8 * s[N_OPEN]) * self._config['MAX_mask'] / s[N_MASK]
       
        self.gamma_mask = min(self.gamma_mask, 1)

    def update_gamma_move(self, s, t):
        """ To update gamma_move in two different way according to timestep.

        Args:
            s : global state
            t : timestep

        """
        if t <= self._config['early_threshold']:

            self.gamma_move = s[N_OPEN] + 1

        else:

            self.gamma_move = self.move_b - self.move_w * s[N_OPEN]

            self.gamma_move = max(0, self.gamma_move)

    def update_history(self, t, update_list=['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']):
        """ To update history including seir model and other states.

        Args:
            s : global state
            t : timestep

        """

        self._history['time'].append(t)

        for name in update_list:

            self._history[name].append(getattr(self, name) / self._config['N_total'])
    
    def plot_history(self, plot_list=['S', 'E', 'Q', 'I', 'R'], out_path='history.png'):
        """ To plot people transmission history line chart.

        Args:
            plot_list:  plot the attributes in this class.
                        please make sure you have store the attributes in update_history
            out_path: save figure to the path.

        """
        display_config = {
            'S': ('Susceptible Population', 'blue'),
            'E': ('Exposed Population', 'orange'),
            'Q': ('Quarantine Population', 'cyan'),
            'I': ('Infectious Population', 'green'),
            'R': ('Recovered Population', 'red'),
            'E_move': ('Exposed Population(move)', 'magenta'),
            'Q_move': ('Quarantine Population(move)', 'black'),
        }

        if any(name not in display_config for name in plot_list):

            raise Exception('Please check each element in plot list must be set in displaying config.')

        fig = plt.figure(out_path)

        history_fig = fig.add_subplot(111)

        for name in plot_list:

            history_fig.plot('time', 'number', '-', color=display_config[name][1], data={
                'time': self._history['time'],
                'number': self._history[name]})

        # history_fig.legend(['Susceptible Population', 'Exposed Population', 'Infectious Population', 'Recovered Population'])
        history_fig.legend([display_config[name][0] for name in plot_list], loc='center right')

        history_fig.set_ylabel('population')

        history_fig.set_xlabel('time')

        history_fig.set_title('SEIR')

        fig.savefig(out_path)

    def print_state(self, t, print_list=['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']):

        log = f'{t}\t'

        log += '\t'.join([f'{name}: {getattr(self, name)}' for name in print_list])

        print(log)
    
    @property
    def start(self):
        return self._start


class Agent(nn.Module):
    """ The decision-making policy network for the simulation.
    Attributes
    ----------
        init_legal_actions: make all actions legal
        legal_actions: legal actions in current state
        log_probs: the log(probability) at every time step
        rewards: RL rewards
        action_embeddings: the transform of the actions
        net: policy network
        cooldown_criteria: CD criteria of the actions. This should be pre-defined when creating an Agent object
        current_cooldown: CD time at current time step. This should be initialized to all zero value
    Methods
    -------
        init_agent: reset the history fo the log probability and rewards
        forward: the inference of the policy network
        select_actions: return the action sampled from the policy nwetwork
        get_legal_actions: update the legal_actions at current state
        update_cooldown : update the cooldown time every time step
    """
    def __init__(self, dim_input, dim_output, max_actions, init_legal_actions, cooldown_criteria):
        """ Initialize the parameters of the policynwtwork
        Args
        ----
            dim_input: the size of the observed state
            dim_output: the size of the state encoding output
            max_actions: len(init_legal_actions)
            init_legal_actions: make all actions legal
            cooldown_criteria: CD criteria of the actions
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

        self.cooldown_criteria = cooldown_criteria

        self.current_cooldown = None

    def init_agent(self):

        self.legal_actions = self.init_legal_actions.copy()

        self.log_probs = []

        self.rewards = []
        
        self.current_cooldown = {}

        for act in self.cooldown_criteria.keys():
            self.current_cooldown[act] = 0
          

    def forward(self, state):

        self.get_legal_actions(state)

        state_vector = self.net(state)

        actions = torch.tensor(self.legal_actions, device=device)

        actions_vectors = self.action_embeddings(actions)

        scores = torch.matmul(actions_vectors, state_vector.unsqueeze(1)).squeeze()

        return F.softmax(scores, dim=0)

    def select_actions(self, scores):

        distribution = Categorical(scores.view(-1, ))

        action = distribution.sample()

        self.log_probs.append(distribution.log_prob(action))

        action_id = self.legal_actions[action.item()]

        self.update_cooldown(action_id)

        return action_id
    
    def get_legal_actions(self, state):

        if state[N_OPEN] >= 0.5 and SET_OPEN in self.legal_actions:

            self.legal_actions.remove(SET_OPEN)

        elif state[N_OPEN] < 0.5 and SET_OPEN not in self.legal_actions:

            self.legal_actions.append(SET_OPEN)

        if state[N_OPEN] >= 1 and SET_OPEN2 in self.legal_actions:

            self.legal_actions.remove(SET_OPEN2)

        elif state[N_OPEN] < 1 and SET_OPEN2 not in self.legal_actions:

            self.legal_actions.append(SET_OPEN2)

        if state[N_OPEN] == 0 and DEC_OPEN in self.legal_actions:

            self.legal_actions.remove(DEC_OPEN)

        elif state[N_OPEN] > 0 and DEC_OPEN not in self.legal_actions:

            self.legal_actions.append(DEC_OPEN)

        for act, cd in self.current_cooldown.items():

            if cd > 0 and act in self.legal_actions:

                self.legal_actions.remove(act)

            elif cd == 0 and act not in self.legal_actions:

                self.legal_actions.append(act)


    def update_cooldown(self, action_id):

        for act, cd in self.current_cooldown.items():

            if action_id != act:

                self.current_cooldown[act] = max(0, cd-1)
            
            else:

                self.current_cooldown[action_id] = self.cooldown_criteria[action_id]

class GRUAgent(Agent):

    def __init__(self, dim_input, dim_output, max_actions, init_legal_actions, cooldown_criteria):

        super().__init__(dim_input, dim_output, max_actions, init_legal_actions, cooldown_criteria)

        self.gru = nn.GRU(dim_input, dim_input, batch_first=True)

        self.memory = None

    def init_agent(self):

        self.legal_actions = self.init_legal_actions.copy()

        self.log_probs = []

        self.rewards = []
        
        self.current_cooldown = {}

        for act in self.cooldown_criteria.keys():
            self.current_cooldown[act] = 0

        self.memory = None

    def forward(self, state):

        self.get_legal_actions(state)

        _, self.memory = self.gru(state.view(1,1,-1), self.memory)

        state_vector = self.net(self.memory.view(-1))

        actions = torch.tensor(self.legal_actions, device=device)

        actions_vectors = self.action_embeddings(actions)

        scores = torch.matmul(actions_vectors, state_vector.unsqueeze(1)).squeeze()

        return F.softmax(scores, dim=0)



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

    def episodes(self, max_episodes=5, max_steps=100, plot=False):

        for episode in range(max_episodes):

            self.agent.init_agent()

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

            if plot:

                self.environment.plot_history(out_path=f'ep{episode:02d}_history.png')

def main():

    init_legal_actions = [TRADE_MASK, SET_OPEN, SET_OPEN2, DEC_OPEN, SWITCH_SHUTDOWN, SWITCH_MOVE_CONTROL, NO_ACTION]

    cooldown_criteria = {TRADE_MASK: 7,
                           SET_OPEN: 3,
                          SET_OPEN2: 3,
                           DEC_OPEN: 3,
                    SWITCH_SHUTDOWN: 30,
                SWITCH_MOVE_CONTROL: 30}

    args_agent = {
        "dim_input": 5, 
        "dim_output": 3,
        "max_actions": 7,
        "init_legal_actions": init_legal_actions,
        "cooldown_criteria": cooldown_criteria
    }

    #agent = Agent(**args_agent).to(device)
    agent = GRUAgent(**args_agent).to(device)

    env = Environment()

    game = Simulatoin(agent, env, optim.Adam(agent.parameters(), lr=1e-3))

    game.episodes(plot=True)


if __name__ == "__main__":

    main()