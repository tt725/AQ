import copy
import time
import json

import agents
from utils.helper import *


class Experiment(object):
  def __init__(self, cfg):
    self.results = {'Train': None, 'Test': None}
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cuda'
      self.device = 'cuda'
    self.config_idx = cfg['config_idx']
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    self.log_path = {'Train': self.cfg['train_log_path'], 'Test': self.cfg['test_log_path']}
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()

  def run(self):
    set_one_thread()
    self.start_time = time.time()
    set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    self.agent.env.seed(self.cfg['seed'])
    self.results['Train'], self.results['Test'] = self.agent.run_steps(render=self.cfg['render'])
    self.save_results(mode='Train')
    self.save_results(mode='Test')
    self.save_model()
    self.end_time = time.time()
    self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')

  def save_results(self, mode):
    self.results[mode]['Env'] = self.results[mode]['Env'].astype('category')
    self.results[mode]['Agent'] = self.results[mode]['Agent'].astype('category')
    self.results[mode].to_feather(self.log_path[mode])
  
  def save_model(self):
    self.agent.save_model(self.model_path)
  
  def load_model(self):
    self.agent.load_model(self.model_path)

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()