# Overview

This repository provides reproducible code and configurations for all experiments reported in our paper ("Revisiting Overestimation Bias of Q-learning: Breaking Bias Propagation Chains Does Well"). 

[ICML2026-AdaAQ-Code](https://1drv.ms/f/c/f6620e9a58a75dd7/IgCpDqi8aLcWTYMdWrTW6X2gAeE30xFtQBOgLQHnOCaDcpA?e=oLpRZH)

Note that the above link includes our all experiment code and data.



## Hardware and Software

```
1.Compute resource:
-- Ubantu: 22.04.5 LTS
-- CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
-- GPU: A100 * 8
-- NVIDIA-SMI: 550.127.05
-- CUDA: 12.4

2.Software resouce
-- Platform: PyCharm2019.3
-- Language: Python3.8
```





## Example Experiments

```
Example/
├── Env.py (multi-arms bandit)
├── Q-learner.py (Q-learning)
├── QMain.py (run)
├── Result/  (experiment result)
├── Figure/  (example figure)
└── requirements.txt (Running packages)  
```



## Table Experiments

```
TableExp/
├── Multiarms\ (include code and data)
├── Roulette\ (include code and data)
├── Gridworld3\ (include code and data)
├── Gridworld4\ (include code and data)
├── Figure/  (table experiment figures)
└── requirements.txt (running packages)  
```



## Discrete-Action DRL Experiments

```

├── DeepExp/
    ├── logs (inculdes experiment results)
    ├── figure (inculdes experiment figures)
    ├── agents 
        ├── AlterDQN.py (our Alternating DQN)
        └── SoftAlterDQN.py (our Adaptive Alternating DQN)
    ├── configs/ (running) 
    └── requirements.txt (running packages) 
  
```



## Continuous-Action DRL Experiments

```
ContinualExp/
├── data (inculdes experiment results)
├── figure (inculdes all experiment figures)
├── experiments 
    ├── train_AQ.py (our Alternating DDPG)
    └── train_SoftAQ.py (our Adaptive Alternating DDPG)
└── requirements.txt
```



