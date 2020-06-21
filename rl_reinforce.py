import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import json
import math
import numpy as np
import collections
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(666)
#torch.manual_seed(666)

from constant import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class Environment:

    """A epidemic infection simulation model.
    Attributes
    ----------
        is_terminal: 
        _start: 
        _history: store all records including seir model or other states
        _warning_threshold: warning threshold
        _danger_threshold: danger threshold should larger than warning one.
                
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
        gamma_detect_move: the factor of mask trade to affect COVID detection(move)
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
    def __init__(self, config_path='config.json', immediate_reward=True, gamma=0.99):

        self._config = self._load_config(config_path)

        self._start = torch.zeros(8).to(device)

        self.is_terminal = False

        self._history = collections.defaultdict(list)

        self._warning_threshold = 0.25

        self._danger_threshold = 0.75

        self._gamma = gamma

        self._immediate_reward = immediate_reward
    
    def _load_config(self, config_path):

        """ Load the environment configuration.
        Returns:
            The dictionary of environment configuration.
        """
        try:

            config = json.load(open(config_path))

        except:

            raise Exception('Failed to load environment config from \'config.json\'')

        return config
    
    def init_state(self, test=False):
        
        """ We need to define this function to initialize the environment.
        Returns:
            The starting state of the environment
        """
        if test:
            
            self._config = self._load_config(config_path='config_test.json')

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
        self.gamma_detect_move = self._config['gamma_detect_move']
        self.gamma_move = self._config['gamma_move']
        self.gamma_shut = self._config['gamma_shut']

        # init terminal state
        self.is_terminal = False

        # init gamma move parameter
        self.move_w = 0
        self.move_b = 0

        self._history = collections.defaultdict(list)

        self._assert_all(if_config=True, if_seir=True)

        self.E_L, self.E_R, self.E_MAX, self.E_t = None, None, float('-inf'), -1
        self.I_L, self.I_R, self.I_MAX, self.I_t = None, None, float('-inf'), -1

        return self.start.clone()
   
    def step(self, s, a, t, test=False):
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

        _origin_immediate_reward = self.get_immediate_reward(s)

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

            s[N_GOLD] = s[N_GOLD] - self._config['shut_rate']

        else:

            self.gamma_shut = 1

            if s[N_GOLD] < self._config['N_gold']:

                s[N_GOLD] = min(self._config['N_gold'], s[N_GOLD] + 0.03 * self._config['shut_rate'])


        # update number of gold by determining if it is move control
        if s[IF_MOVE_CONTROL] == 1:

            s[N_GOLD] = s[N_GOLD] - self._config['inc_move']

        else:

            if s[N_GOLD] < self._config['N_gold']:

                s[N_GOLD] = min(self._config['N_gold'], s[N_GOLD] + 0.03 * self._config['inc_move'])


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

                self.gamma_detect = min(0.95 / self._config['alpha_eq'], self.gamma_detect)

                self.gamma_detect_move += self._config['inc_detect']

                self.gamma_detect_move = min(0.95 / self._config['alpha_eq_move'], self.gamma_detect_move)

        elif a == SET_OPEN:

            s[N_OPEN] = 0.5
        
        elif a == SET_OPEN2:

            s[N_OPEN] = 1.0
        
        elif a == DEC_OPEN:

            s[N_OPEN] -= 0.1
            
            s[N_OPEN] = max(0, np.around(s[N_OPEN].item(), decimals=1)) # should be larger than 0

        elif a == SWITCH_SHUTDOWN:
            
            s[IF_SHUTDOWN] = 1 - s[IF_SHUTDOWN]
       
        elif a == SWITCH_MOVE_CONTROL:

            s[IF_MOVE_CONTROL] = 1 - s[IF_MOVE_CONTROL]

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

            EQ_move = np.random.binomial(self.E_move - EI_move, self.gamma_detect_move * self._config['alpha_eq_move'])
            
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

        if self.E + self.E_move > self.E_MAX:

            self.E_MAX = self.E + self.E_move

            self.E_t = t

        if self.I > self.I_MAX:

            self.I_MAX = self.I

            self.I_t = t
        
        s[N_QUARANTINE] = self.Q + self.Q_move

        """ update reward """
        
        immediat_reward = 0.

        if self._immediate_reward:

            immediat_reward = self._gamma * self.get_immediate_reward(s) - _origin_immediate_reward

        terminal_reward = 0.
        
        if self.R / self._config['N_total'] > 0.9 and not test:

            terminal_reward = self.evaluation()

            self.is_terminal = True

        reward = (terminal_reward, immediat_reward)
        
        self._assert_all(s=s, if_seir=True, if_variable=True)

        self.update_history(t, a, s[N_GOLD].item(), sum(reward))

        return s, reward, self.is_terminal

    def get_immediate_reward(self, state):

        score = state[N_GOLD] - (self.E + self.I * 2) / self._config['N_total']

        return score

    def evaluation(self):
        
        score_e = 0.

        if self.E_MAX / self._config['N_total'] < self._warning_threshold:
        
            score_e += 10
        
        elif self.E_MAX / self._config['N_total'] < self._danger_threshold:

            score_e += 5
        
        else:

            score_e -= 10
        
        # if self.I_MAX / self._config['N_total'] < self._warning_threshold:
        
        #     score += 2
        
        # elif self.I_MAX / self._config['N_total'] < self._danger_threshold:

        #     score += 1
        
        # else:

        #     score -= 10

        return score_e

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

            gamma_list = ['gamma_mask', 'gamma_move', 'gamma_shut', 'gamma_recover', 'gamma_detect', 'gamma_detect_move']

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
            
            assert 0 <= self.gamma_shut <= 1 # TODO

            assert 0 <= self.gamma_recover * self._config['alpha_ir'] <= 1 # TODO

            assert 0 <= self.gamma_detect * self._config['alpha_eq'] <= 1 # TODO

            assert 0 <= self.gamma_detect_move * self._config['alpha_eq_move'] <= 1 # TODO

        if if_seir or if_all:

            pop_rule = lambda x: 0 <= x <= self._config['N_total']

            for name in ['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']:

                assert pop_rule(getattr(self, name))

            assert self.S + self.E + self.E_move + self.Q + self.Q_move + self.I + self.R == self._config['N_total']
            
            assert 0 <= self.Q + self.Q_move < self._config['MAX_Q']

    def obeserved_state(self, state):
        """ To transform the global state to the obsevable state for agent.
        Args:
            state : global state
     
        Returns:
            observed_state : obsevable state for agent
        """
        observed_state = torch.zeros(7).to(device)
        observed_state[:5] = state[:5].clone()
        observed_state[5] = self.I
        observed_state[6] = self.R
        return observed_state
    
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

    def update_history(self, t, action, gold, reward, update_list=['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']):
        """ To update history including seir model and other states.
        Args:
            s : global state
            t : timestep
        """

        self._history['time'].append(t)

        self._history['action'].append(action)

        self._history['gold'].append(gold)

        self._history['reward'].append(reward)

        for name in update_list:
            if name == 'Q' or name == 'Q_move':

                self._history[name].append(getattr(self, name) / self._config['MAX_Q'])

            else: 

                self._history[name].append(getattr(self, name) / self._config['N_total'])
    
    def plot_history(self, plot_list=['S', 'E', 'I', 'R', 'Q', 'Q_move', 'gold'], out_path='history.png', truncate=-1, staggered=True, annotate_action=False):
        """ To plot people transmission history line chart.
        Args:
            plot_list:  plot the attributes in this class.
                        please make sure you have store the attributes in update_history
            out_path: save figure to the path.
        """
        display_config = {
            'S': ('Susceptible', 'blue'),
            'E': ('Exposed', 'orange'),
            'Q': ('Quarantine', 'cyan'),
            'I': ('Infectious', 'green'),
            'R': ('Recovered', 'red'),
            'E_move': ('Exposed(move)', 'magenta'),
            'Q_move': ('Quarantine(move)', 'brown'),
            'reward': ('Reward', 'magenta'),
            'gold': ('Gold', 'black'),
        }

        if any(name not in display_config for name in plot_list):

            raise Exception('Please check each element in plot list must be set in displaying config.')

        if annotate_action:

            figsize = (15, 5)
        
        else:

            figsize = (6, 4)

        fig = plt.figure(out_path, figsize=figsize, dpi=400)

        history_fig = fig.add_subplot(111)

        for name in plot_list:

            history_fig.plot('time', 'number', '-', color=display_config[name][1], data={
                'time': self._history['time'][:truncate],
                'number': self._history[name][:truncate]})
        
        plt.axhline(y=self._danger_threshold, color='r', linestyle='--')

        plt.axhline(y=self._warning_threshold, color='orange', linestyle='--')

        # history_fig.legend(['Susceptible Population', 'Exposed Population', 'Infectious Population', 'Recovered Population'])
        history_fig.legend([display_config[name][0] for name in plot_list], loc='center right')

        history_fig.set_ylabel('population')

        history_fig.set_xlabel('time')

        history_fig.set_title('SEIR')

        if annotate_action:

            flag = 1

            for t, action in enumerate(self._history['action'][:truncate]):

                if action != NO_ACTION:

                    if flag == 1 or not staggered:
                    
                        history_fig.text(t, -0.05, ACTIONS[action], fontsize=7, rotation=45, verticalalignment='bottom', horizontalalignment='left')
                    
                    else:
                    
                        history_fig.text(t, -0.05, ACTIONS[action], fontsize=7, rotation=-45, verticalalignment='top', horizontalalignment='left')
                    
                    flag = 1 - flag

        fig.savefig(out_path)

        plt.close(fig)

    def print_state(self, t, print_list=['S', 'E', 'E_move', 'Q', 'Q_move', 'I', 'R']):

        log = f'{t:3d} '

        log += '\t'.join([f'{name}: {int(getattr(self, name)):>8d}' for name in print_list])

        log += f'\tsc: {self.evaluation():>4.2f}'

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
        init_weight : initialize the weights of the model with given distribution.
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
            nn.Linear(dim_input, dim_input),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(dim_input, dim_output)
        )

        self.cooldown_criteria = cooldown_criteria

        self.current_cooldown = None

    def init_weight(self, init_func, *params, **kwargs):

        for p in self.parameters():
            
            init_func(p, *params, **kwargs)

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

    def greedy_actions(self, scores):

        action = torch.argmax(scores.detach())

        action_id = self.legal_actions[action.item()]

        self.update_cooldown(action_id)

        return action_id  
    
    def get_legal_actions(self, state):

        
        freeze_buffer = set()

        if state[N_OPEN] >= 0.5:
            
            if  SET_OPEN in self.legal_actions:

                self.legal_actions.remove(SET_OPEN)

            freeze_buffer.add(SET_OPEN)

        elif SET_OPEN not in self.legal_actions:

            self.legal_actions.append(SET_OPEN)

        if state[N_OPEN] >= 1:
            
            if  SET_OPEN2 in self.legal_actions:

                self.legal_actions.remove(SET_OPEN2)

            freeze_buffer.add(SET_OPEN2)

        elif SET_OPEN2 not in self.legal_actions:

            self.legal_actions.append(SET_OPEN2)

        if state[N_OPEN] == 0:
            
            if DEC_OPEN in self.legal_actions:

                self.legal_actions.remove(DEC_OPEN)

            freeze_buffer.add(DEC_OPEN)

        elif state[N_OPEN] > 0 and DEC_OPEN not in self.legal_actions:

            self.legal_actions.append(DEC_OPEN)
        
        for act, cd in self.current_cooldown.items():

            if cd > 0:

                if act in self.legal_actions:

                    self.legal_actions.remove(act)

            elif act not in self.legal_actions and act not in freeze_buffer:

                self.legal_actions.append(act)

        assert len(self.legal_actions) == len(set(self.legal_actions)), "Redundant Actions"

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

        super().init_agent()

        self.memory = None

    def forward(self, state):

        self.get_legal_actions(state)

        _, self.memory = self.gru(state.view(1,1,-1), self.memory)

        state_vector = self.net(self.memory.view(-1))

        actions = torch.tensor(self.legal_actions, device=device)

        actions_vectors = self.action_embeddings(actions)

        scores = torch.matmul(actions_vectors, state_vector.unsqueeze(1)).squeeze()

        return F.softmax(scores, dim=0)


class ActorCriticAgent(GRUAgent):

    def __init__(self, dim_input, dim_output, max_actions, init_legal_actions, cooldown_criteria):

        super().__init__(dim_input, dim_output, max_actions, init_legal_actions, cooldown_criteria)

        self.value_gru = nn.GRU(dim_input, dim_input, batch_first=True)

        self.value_net = nn.Sequential(
            nn.Linear(dim_input, 1)
        )

        self.state_values = []

        self.memory_value = None

    def init_agent(self):

        super().init_agent()

        self.state_values = []

        self.memory_value = None

    def forward_value(self, state):

        _, self.memory_value = self.gru(state.view(1,1,-1), self.memory_value)

        state_value = self.value_net(self.memory_value.view(-1))

        return state_value

    def select_actions(self, scores, state_value):

        action_id = super().select_actions(scores)

        self.state_values.append(state_value)

        return action_id


class Simulatoin:
    """ The interaction between environment(epidemic infection model) and agent(policy network)
    Attributes
    ----------
        agent
        enviroment
        gamma
        optimizer
        verbose
        baseline
        is_actor_critic
    Methods
    -------
        testing: tesintg mode
        get_baseline: get the naive sampling-based baseline of accumulated reward (still under development)
        episodic_policy_loss:
        episodes:
        monti_carlo_estimation:
        save_agent:
        load_agent:
    """

    def __init__(self, agent: Agent, baseline_agent: Agent, environment: Environment, optimizer=None, gamma=0.9, verbose=False):

        self.agent = agent

        self.baseline_agent = baseline_agent

        self.environment = environment

        self.testing_environment = Environment(config_path='config_test.json', gamma=gamma)

        self.gamma = gamma

        self.optimizer = optimizer

        self.verbose = verbose

        self.baseline = None

        self.is_actor_critic = False

    def testing(self, max_steps=300, max_episode=1, greedy=True, load=False, plot=True, verbose=True, baseline=False):

        assert (load and not baseline) or (not load), "If we set the baseline to True and load to True, the loaded agent will not be tested."

        if load:

            self.load_agent()

        if baseline:

            agent = self.baseline_agent

        else:

            agent = self.agent

        agent.eval()

        reward_total = 0

        score_total = 0

        actions_list = []

        for ep in range(max_episode):

            agent.init_agent()

            state = self.testing_environment.init_state(test=True)

            state_observed = self.testing_environment.obeserved_state(state)

            reward_eps = 0

            actions_list.append([])

            for t in range(max_steps):

                actions_probs = agent.forward(state_observed)

                if greedy:

                    action = agent.greedy_actions(actions_probs)


                else:

                    if self.is_actor_critic:
                        
                        state_values = agent.forward_value(state_observed)

                        action = agent.select_actions(actions_probs, state_values)


                    else:

                        action = agent.select_actions(actions_probs)
                '''
                if t <=2:
                    print(actions_probs)
                    print(action)
                else:
                    exit()
                '''
                
                state, reward, is_terminal = self.testing_environment.step(state, action, t=t, test=True)

                reward_eps += sum(reward)

                state_observed = self.testing_environment.obeserved_state(state)

                if is_terminal:

                    break

                actions_list[-1].append(action)
            
            if ep % 10 == 0 and plot:

                if load == False:

                    out_path = '(INIT) '
                
                else:

                    out_path = '(TEST) '

                out_path += f'ep{ep:02d}_history.png'

                self.testing_environment.plot_history(plot_list=["E", "I", "gold"], out_path=out_path, truncate=200, annotate_action=True)

            reward_total += reward_eps

            score_total += self.testing_environment.evaluation() + self.testing_environment.get_immediate_reward(state)

        reward_avg = reward_total / max_episode

        score_avg = score_total / max_episode

        if verbose:

            print("(Testing) Avg Reward {:>6.2f} | Avg Score {:>6.2f}".format(reward_avg, score_avg))

        agent.train()

        return reward_avg, score_avg, actions_list

    def get_baseline(self, max_episodes=50, max_steps=400):

        R = {}

        for step in range(max_steps):

            R[step] = []

        for _ in range(max_episodes):

            state = self.environment.init_state()

            rewards = []

            for t in range(max_steps):

                action = NO_ACTION
                
                state, reward, is_terminal = self.environment.step(state, action, t=t)

                rewards.append(sum(reward))

                if is_terminal:

                    break

            v = 0

            gamma = self.gamma*0

            accumulated_rewards = []

            for r in rewards[::-1]:   # r_T, r_{T-1}, .....r_0

                v = r + gamma * v

                accumulated_rewards.insert(0, v)

            for i, ar in enumerate(accumulated_rewards):

                ar_normalize = (ar - np.mean(accumulated_rewards)) / (np.std(accumulated_rewards) + 1e-9) 

                R[i].append(ar_normalize)

        for k, v in R.items():
            
            if len(v) > 0:

                R[k] = np.mean(v)

            else:

                R[k] = 0

        self.baseline = R

    def episodic_policy_loss(self):

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

        for i , (log_prob, R) in enumerate(zip(self.agent.log_probs, accumulated_rewards)):

            #policy_loss += (-log_prob * (R - self.baseline[i]))
            policy_loss += (-log_prob * R)
        
        return policy_loss

    def actor_critic_policy_loss(self):

        policy_loss = 0

        value_loss = 0

        accumulated_rewards_terminal, accumulated_rewards_immediate = [], []

        # v_t = r_t + gammar*v_{t+1}
        # v_T = 0
        # agent.rewards : [r_0, r_1, r_2, ........ r_T]

        v_terminal, v_immediate = 0, 0   # V_T = 0

        v_terminal_T, v_immediate_T = self.agent.rewards[-1]

        t = 0

        for r_terminal, r_immediate in self.agent.rewards[::-1]:   # r_T, r_{T-1}, .....r_0

            if t < 300 - self.environment.E_t:

                v_terminal = 0 * r_terminal

                v_immediate = 0 * r_immediate
            
            elif t == 300 - self.environment.E_t:

                v_terminal = v_terminal_T

                v_immediate = v_immediate_T
            
            else:
            
                v_terminal = r_terminal + self.gamma * v_terminal

                v_immediate = r_immediate + self.gamma * v_immediate

            accumulated_rewards_terminal.insert(0, v_terminal)

            accumulated_rewards_immediate.insert(0, v_immediate)
            
            t += 1

        accumulated_rewards_terminal = torch.tensor(accumulated_rewards_terminal, device=device)

        accumulated_rewards_immediate = torch.tensor(accumulated_rewards_immediate, device=device)

        #accumulated_rewards_terminal = (accumulated_rewards_terminal - accumulated_rewards_terminal.mean()) / (accumulated_rewards_terminal.std() + 1e-9)
        
        #accumulated_rewards_immediate = (accumulated_rewards_immediate - accumulated_rewards_immediate.mean()) / (accumulated_rewards_immediate.std() + 1e-9)

        t = 0
        
        for log_prob, (R_terminal, R_immediate), value in zip(self.agent.log_probs, zip(accumulated_rewards_terminal, accumulated_rewards_immediate), self.agent.state_values):
            
            if t < self.environment.E_t:


                policy_loss += (-log_prob * ((R_terminal + R_immediate) - value.item()))

                value_loss += F.smooth_l1_loss(value, torch.tensor([R_terminal + R_immediate], device=device))

            t += 1

        loss = policy_loss + value_loss

        return loss

    def episodes(self, max_episodes=50, max_steps=400, early_stop=100, plot=False):

        local_best = float('-inf')

        early_stop_flag = 0

        for episode in range(max_episodes):

            self.agent.init_agent()

            state = self.environment.init_state()

            state_observed = self.environment.obeserved_state(state)

            loss_eps = 0

            reward_eps = 0

            score_eps = 0

            for t in range(max_steps):

                if self.is_actor_critic:

                    actions_probs = self.agent.forward(state_observed)

                    state_values = self.agent.forward_value(state_observed)

                    action = self.agent.select_actions(actions_probs, state_values)

                else:

                    actions_probs = self.agent.forward(state_observed)

                    action = self.agent.select_actions(actions_probs)

                state, reward, is_terminal = self.environment.step(state, action, t=t)

                reward_eps += sum(reward)

                state_observed = self.environment.obeserved_state(state)

                self.agent.rewards.append(reward)

                if is_terminal:

                    break
            
            score_eps += self.environment.evaluation()

            if self.is_actor_critic:

                loss_eps = self.actor_critic_policy_loss()

            else:

                loss_eps = self.episodic_policy_loss()

            _, test_score, _ = self.testing(max_steps=max_steps, greedy=True, plot=False, verbose=False)

            if self.verbose:

                print("Episode {:>3d} | Loss {:>6.2f} | Train Score {:>6.2f} | Test Score {:>6.2f}".format(episode, loss_eps.item(), score_eps, test_score))

            self.optimizer.zero_grad()
            
            loss_eps.backward()

            self.optimizer.step()

            if test_score > local_best:

                self.save_agent()

                local_best = test_score

                early_stop_flag = 0
            
            elif early_stop_flag >= early_stop:

                break

            else:

                early_stop_flag += 1

            if plot:

                self.environment.plot_history(out_path=f'ep{episode:02d}_history.png')

    def monti_carlo_estimation(self, iterations=3, num_rollouts=5, max_steps=400, early_stop=30, plot=False):

        local_best = float('-inf')
        
        early_stop_flag = 0

        for i in range(iterations):

            loss_i = 0

            train_reward = 0

            train_score = 0

            for _ in range(num_rollouts):

                self.agent.init_agent()

                state = self.environment.init_state()

                state_observed = self.environment.obeserved_state(state)

                for t in range(max_steps):

                    if self.is_actor_critic:

                        actions_probs = self.agent.forward(state_observed)

                        state_values = self.agent.forward_value(state_observed)

                        action = self.agent.select_actions(actions_probs, state_values)

                    else:

                        actions_probs = self.agent.forward(state_observed)

                        action = self.agent.select_actions(actions_probs)
  
                    state, reward, is_terminal = self.environment.step(state, action, t=t)

                    state_observed = self.environment.obeserved_state(state)

                    self.agent.rewards.append(reward)

                    train_reward += sum(reward)

                    if is_terminal: 
                        
                        break

                if self.is_actor_critic:

                    loss_i += self.actor_critic_policy_loss()

                else:

                    loss_i += self.episodic_policy_loss()

                train_score += self.environment.evaluation() + self.environment.get_immediate_reward(state)

            test_reward, test_score, _ = self.testing(max_steps=max_steps, max_episode=10, greedy=True, plot=False, verbose=False)

            loss_i /= num_rollouts

            train_reward /= num_rollouts

            train_score /= num_rollouts

            if test_score > local_best:

                self.save_agent()

                local_best = test_score

                early_stop_flag = 0
            
            elif early_stop_flag >= early_stop:

                break

            else:

                early_stop_flag += 1

            self.optimizer.zero_grad()

            if self.verbose:

                print("(MC) Iteration {:>3d} | Avg Loss {:>6.2f} | Train Reward/Score {:>6.2f}/{:>6.2f} | Test Reward/Score {:>6.2f}/{:>6.2f}".format(i, loss_i.item(), train_reward, train_score, test_reward, test_score))

            loss_i.backward()

            self.optimizer.step()

            if plot:

                self.environment.plot_history(out_path=f'(MC) iteration{i:02d}_history.png')

    def save_agent(self, out_path='./save/model.pt'):

        torch.save(self.agent, out_path)

    def load_agent(self, in_path='./save/model.pt'):

        self.agent = torch.load(in_path)           
            

def main():

    init_legal_actions = [TRADE_MASK, SET_OPEN, SET_OPEN2, DEC_OPEN, SWITCH_SHUTDOWN, SWITCH_MOVE_CONTROL, NO_ACTION]

    cooldown_criteria = {TRADE_MASK: 7-5,
                           SET_OPEN: 3-1,
                          SET_OPEN2: 3-1,
                           DEC_OPEN: 3-1,
                    SWITCH_SHUTDOWN: 30-25,
                SWITCH_MOVE_CONTROL: 30-25}

    args_agent = {
        "dim_input": 7, 
        "dim_output": 7,
        "max_actions": 7,
        "init_legal_actions": init_legal_actions,
        "cooldown_criteria": cooldown_criteria
    }

    agent = ActorCriticAgent(**args_agent).to(device)

    baseline_agent = ActorCriticAgent(**args_agent).to(device)

    agent.init_weight(torch.nn.init.normal_, mean=0., std=0.01)

    baseline_agent.init_weight(torch.nn.init.zeros_)

    env = Environment(gamma=0.9)

    optimizer = optim.Adam(agent.parameters(), lr=5e-2)

    game = Simulatoin(agent, baseline_agent, env, optimizer, verbose=True, gamma=0.9)

    game.is_actor_critic = True

    #r_0, s_0, a_0 = game.testing(max_steps=300, max_episode=50, greedy=False, baseline=True) # Random walk
    r_0, s_0, a_0 = game.testing(max_steps=300, max_episode=50, greedy=True, baseline=True) # No action

    try:
        #game.episodes(max_episodes=1000, max_steps=200, plot=False)
        game.monti_carlo_estimation(iterations=100, num_rollouts=5, max_steps=300, plot=False)
    
    except KeyboardInterrupt:

        pass

    r_1, s_1, a_1 = game.testing(max_steps=300, max_episode=50, greedy=True, load=True)

    # Before training
    print("Iint Model actions taken")

    for n in range(3): # print 3 rollouts of actions

        print(a_0[n][:50]) # action taken in time step 0 ~ 49


    # After training

    print("Trained Model actions taken")

    for n in range(3): # print 3 rollouts of actions 

        print(a_1[n][:50]) # actions taken in time step 0 ~ 49


    print("Initi Model Reward {:>6.2f} Score {:>6.2f}".format(r_0, s_0))

    print("Train Model Reward {:>6.2f} Score {:>6.2f}".format(r_1, s_1))


if __name__ == "__main__":

    main()