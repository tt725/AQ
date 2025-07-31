from utils.helper import *
from agents.VanillaDQN import VanillaDQN


class SoftAlterDQN(VanillaDQN):
  '''
  Implementation of SoftAlterDQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.soft_parameter = cfg['agent']['soft_parameter']
    self.target_network_update_freqency = cfg['target_network_update_freqency']
    self.Q_net_target = [None]
    self.Q_net_target[0] = self.creatNN(cfg['env']['input_type']).to(self.device)
    self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
    self.Q_net_target[0].eval()

  def learn(self):
    super().learn()
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

  def compute_q_target(self, next_states, rewards, dones):
    q_next = self.Q_net_target[0](next_states).detach().max(1)[0]

    best_actions = self.Q_net[0](next_states).detach().argmax(1).unsqueeze(1)
    dq_next = self.Q_net_target[0](next_states).detach().gather(1, best_actions).squeeze()

    q = torch.zeros(size=[self.batch_size]).to(self.device)
    for i in range(0, self.batch_size):
        if torch.exp(self.soft_parameter*abs(dq_next[i])).to(self.device) / (torch.exp(self.soft_parameter*abs(dq_next[i])).to(self.device)+torch.exp(self.soft_parameter*abs(q_next[i])).to(self.device))>torch.rand(1).to(self.device):
            q[i] = q_next[i]
        else:
            q[i] = dq_next[i]

    q_target = rewards + self.discount * q * (1 - dones)

    return q_target