import pandas as pd
from envs.env import *
from utils.helper import *
from components.network import *
from components.normalizer import *
import components.replay
import components.exploration
from agents.BaseAgent import BaseAgent


class VanillaDQN(BaseAgent):
    '''
  Implementation of Vanilla DQN with only replay buffer (no target network)
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.env_name = cfg['env']['name']
        self.agent_name = cfg['agent']['name']
        self.max_episode_steps = int(cfg['env']['max_episode_steps'])
        self.env = make_env(cfg['env']['name'], max_episode_steps=self.max_episode_steps)
        self.config_idx = cfg['config_idx']
        self.device = cfg['device']
        self.batch_size = cfg['batch_size']
        self.discount = cfg['discount']
        self.train_steps = int(cfg['env']['train_steps'])
        self.test_per_episodes = int(cfg['env']['test_per_episodes'])
        self.display_interval = cfg['display_interval']
        self.gradient_clip = cfg['gradient_clip']
        self.action_size = self.get_action_size()
        self.state_size = self.get_state_size()
        self.rolling_score_window = cfg['rolling_score_window']
        if 'MinAtar' in self.env_name:
            self.history_length = self.env.game.state_shape()[2]
        else:
            self.history_length = cfg['history_length']
        self.sgd_update_frequency = cfg['sgd_update_frequency']
        self.show_tb = cfg['show_tb']

        if cfg['env']['input_type'] == 'pixel':
            self.layer_dims = [cfg['feature_dim']] + cfg['hidden_layers'] + [self.action_size]
            if 'MinAtar' in self.env_name:
                self.state_normalizer = RescaleNormalizer()
                self.reward_normalizer = RescaleNormalizer()
            else:
                self.state_normalizer = ImageNormalizer()
                self.reward_normalizer = SignNormalizer()
        elif cfg['env']['input_type'] == 'feature':
            self.layer_dims = [self.state_size] + cfg['hidden_layers'] + [self.action_size]
            self.state_normalizer = RescaleNormalizer()
            self.reward_normalizer = RescaleNormalizer()
        else:
            raise ValueError(f"{cfg['env']['input_type']} is not supported.")
        self.Q_net = [None]
        self.Q_net[0] = self.creatNN(cfg['env']['input_type']).to(self.device)
        self.optimizer = [None]
        self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
                                                                           **cfg['optimizer']['kwargs'])
        self.replay_buffer = getattr(components.replay, cfg['memory_type'])(cfg['memory_size'], self.batch_size,
                                                                            self.device)
        epsilon = {
            'steps': float(cfg['epsilon_steps']),
            'start': cfg['epsilon_start'],
            'end': cfg['epsilon_end'],
            'decay': cfg['epsilon_decay']
        }
        self.exploration_steps = cfg['exploration_steps']
        self.exploration = getattr(components.exploration, cfg['exploration_type'])(cfg['exploration_steps'], epsilon)
        self.loss = getattr(torch.nn, cfg['loss'])(reduction='mean')
        if self.show_tb:
            self.logger.init_writer()
        self.update_Q_net_index = 0

    def creatNN(self, input_type):
        if input_type == 'pixel':
            if 'MinAtar' in self.env_name:
                feature_net = Conv2d_MinAtar(in_channels=self.history_length, feature_dim=self.layer_dims[0])
            else:
                feature_net = Conv2d_Atari(in_channels=self.history_length, feature_dim=self.layer_dims[0])
            value_net = MLP(layer_dims=self.layer_dims, hidden_activation=nn.ReLU())
            NN = NetworkGlue(feature_net, value_net)
        elif input_type == 'feature':
            NN = MLP(layer_dims=self.layer_dims, hidden_activation=nn.ReLU())
        else:
            raise ValueError(f'{input_type} is not supported.')
        return NN

    def reset_game(self):
        self.state = self.state_normalizer(self.env.reset())
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_reward = 0

    def run_steps(self, render=False):
        self.step_count = 0
        self.episode_count = 0
        result = {'Train': [], 'Test': []}
        rolling_score = {'Train': 0.0, 'Test': 0.0}
        total_episode_reward_list = {'Train': [], 'Test': []}
        mode = 'Train'
        while self.step_count < self.train_steps:
            if mode == 'Train' and self.episode_count % self.test_per_episodes == 0:
                mode = 'Test'
            else:
                mode = 'Train'
            self.set_Q_net_mode(mode)
            self.run_episode(mode, render)
            total_episode_reward_list[mode].append(self.total_episode_reward)
            rolling_score[mode] = np.mean(total_episode_reward_list[mode][-1 * self.rolling_score_window[mode]:])
            result_dict = {'Env': self.env_name,
                           'Agent': self.agent_name,
                           'Episode': self.episode_count,
                           'Step': self.step_count,
                           'Return': self.total_episode_reward,
                           'Average Return': rolling_score[mode]}
            result[mode].append(result_dict)
            if self.show_tb:
                self.logger.add_scalar(f'{mode}_Return', self.total_episode_reward, self.step_count)
                self.logger.add_scalar(f'{mode}_Average_Return', rolling_score[mode], self.step_count)
            if self.episode_count % self.display_interval == 0:
                self.logger.info(
                    f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score[mode]:.2f}, Return={self.total_episode_reward:.2f}')

        return pd.DataFrame(result['Train']), pd.DataFrame(result['Test'])

    def run_episode(self, mode, render):
        self.reset_game()
        while not self.done:
            self.action = self.get_action(mode)
            if render:
                self.env.render()
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            self.next_state = self.state_normalizer(self.next_state)
            self.reward = self.reward_normalizer(self.reward)
            if mode == 'Train':
                self.save_experience()
                if self.time_to_learn():
                    self.learn()
                self.step_count += 1
            self.total_episode_reward += self.reward
            self.state = self.next_state
        if mode == 'Train':
            self.episode_count += 1

    def get_action(self, mode='Train'):
        state = to_tensor(self.state, device=self.device)
        state = state.unsqueeze(0)  # Add a batch dimension (Batch, Channel, Height, Width)
        q_values = self.get_action_selection_q_values(state)
        if mode == 'Train':
            action = self.exploration.select_action(q_values, self.step_count)
        elif mode == 'Test':
            action = np.argmax(q_values)  # During test, select best action
        return action

    def time_to_learn(self):
        if self.step_count > self.exploration_steps and self.step_count % self.sgd_update_frequency == 0:
            return True
        else:
            return False

    def learn(self):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample()
        q_target = self.compute_q_target(next_states, rewards, dones)
        q = self.comput_q(states, actions)
        loss = self.loss(q, q_target)
        if self.show_tb:
            self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
        self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
        self.optimizer[self.update_Q_net_index].zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)
        self.optimizer[self.update_Q_net_index].step()

    def compute_q_target(self, next_states, rewards, dones):
        q_next = self.Q_net[0](next_states).detach().max(1)[0]
        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def comput_q(self, states, actions):
        actions = actions.long()
        q = self.Q_net[self.update_Q_net_index](states).gather(1, actions).squeeze()
        return q

    def save_experience(self):
        experience = [self.state, self.action, self.next_state, self.reward, self.done]
        self.replay_buffer.add([experience])

    def get_action_size(self):
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            raise ValueError('Unknown action type.')

    def get_state_size(self):
        return int(np.prod(self.env.observation_space.shape))

    def set_Q_net_mode(self, mode):
        if mode == 'Test':
            for i in range(len(self.Q_net)):
                self.Q_net[i].eval()
        elif mode == 'Train':
            for i in range(len(self.Q_net)):
                self.Q_net[i].train()

    def get_action_selection_q_values(self, state):
        q_values = self.Q_net[0](state)
        q_values = to_numpy(q_values).flatten()
        return q_values

    def save_model(self, model_path):
        state_dicts = {}
        for i in range(len(self.Q_net)):
            state_dicts[i] = self.Q_net[i].state_dict()
        torch.save(state_dicts, model_path)

    def load_model(self, model_path):
        state_dicts = torch.load(model_path)
        for i in range(len(self.Q_net)):
            self.Q_net[i].load_state_dict(state_dicts[i])
            self.Q_net[i] = self.Q_net[i].to(self.device)