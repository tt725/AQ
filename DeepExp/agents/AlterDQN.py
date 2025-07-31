from agents.VanillaDQN import VanillaDQN


class AlterDQN(VanillaDQN):
  '''
  Implementation of AlterDQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.QL_num = cfg['agent']['QL_num']
    self.DQL_num = cfg['agent']['DQL_num']
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
    if self.step_count % (self.QL_num+self.DQL_num) < self.QL_num:
        q_next = self.Q_net_target[0](next_states).detach().max(1)[0]
        q_target = rewards + self.discount * q_next * (1 - dones)
    else:
        best_actions = self.Q_net[0](next_states).detach().argmax(1).unsqueeze(1)
        q_next = self.Q_net_target[0](next_states).detach().gather(1, best_actions).squeeze()
        q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target