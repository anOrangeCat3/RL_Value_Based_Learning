from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random

class ReplayBuffer():
    def __init__(self,capacity:int,device:torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)
        
    def append(self,exp_data:tuple) -> None:
        self.buffer.append(exp_data)
        
    def sample(self,batch_size:int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        mini_batch = random.sample(self.buffer,batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        
        obs_batch = torch.tensor(np.array(obs_batch),dtype=torch.float32,device=self.device)
        
        action_batch = torch.tensor(action_batch,dtype=torch.int64,device=self.device).unsqueeze(1) 
        
        reward_batch = torch.tensor(reward_batch,dtype=torch.float32,device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch),dtype=torch.float32,device=self.device)
        done_batch = torch.tensor(done_batch,dtype=torch.float32,device=self.device)
          
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQN:
    def __init__(self,
                 QNet,
                 action_dim:int,
                 optimizer:torch.optim.Optimizer,
                 replay_buffer:ReplayBuffer,
                 replay_start_size:int,
                 batch_size:int,
                 replay_frequent:int,
                 target_sync_frequent:int, 
                 gamma:float=0.9,
                 epsilon:float=0.1, # Initial epsilon
                 epsilon_decay_rate:float=0.0001, # Decay rate of epsilon
                 mini_epsilon:float=0.01, # Minimum epsilon
                 device:torch.device=torch.device("cpu"),
                 dqn_type:str="DQN"):
        self.action_dim = action_dim
        
        # Initialize the target and Q networks
        self.main_QNet = QNet
        self.target_QNet = copy.deepcopy(QNet)

        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        
        self.replay_start_size = replay_start_size
        self.replay_frequent = replay_frequent
        self.target_sync_frequent = target_sync_frequent
        self.batch_size = batch_size
        self.counter = 0
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate=epsilon_decay_rate
        self.mini_epsilon=mini_epsilon
        self.device = device
        self.dqn_type = dqn_type


    def get_target_action(self, obs:np.ndarray)->int:
        obs=torch.tensor(obs,dtype=torch.float32,device=self.device)
        Q_values=self.target_QNet(obs)
        action=torch.argmax(Q_values).item()
        return action
    
    def get_behavior_action(self, obs:np.ndarray)->int:
        '''
        Here, a simple epsilon decay is used to balance the exploration and exploitation.
        The epsilon is decayed from epsilon_init to mini_epsilon.
        '''
        self.epsilon=max(self.mini_epsilon,self.epsilon-self.epsilon_decay_rate)

        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.get_target_action(obs)

        return action

    def sync_target_QNet(self)->None:
        '''
        Sync the target_QNet with the main_QNet.
        '''
        for target_params, main_params in zip(self.target_QNet.parameters(), self.main_QNet.parameters()):
            target_params.data.copy_(main_params.data)
        
        
    def batch_Q_approximation(self, 
                              batch_obs:torch.Tensor,
                              batch_action:torch.Tensor,
                              batch_reward:torch.Tensor,
                              batch_next_obs:torch.Tensor,
                              batch_done:torch.Tensor)->None:
        '''
        use the main_Q to select the best next action
        use the target_Q to evaluate the Q value of the next action.
        '''
        batch_current_Q = torch.gather(self.main_QNet(batch_obs),1,batch_action).squeeze(1)

        if self.dqn_type == "DQN":
            best_next_action = torch.argmax(self.target_QNet(batch_next_obs),dim=1).unsqueeze(1)
        elif self.dqn_type=="Double_DQN" or self.dqn_type=="Dueling_DQN":
            best_next_action = torch.argmax(self.main_QNet(batch_next_obs),dim=1).unsqueeze(1)

        batch_TD_target = batch_reward + (1-batch_done) * self.gamma * self.target_QNet(batch_next_obs).gather(1,best_next_action).squeeze(1)
        loss = torch.mean(F.mse_loss(batch_current_Q,batch_TD_target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def Q_approximation(self, 
                        obs:np.ndarray,
                        action:int,
                        reward:float,
                        next_obs:np.ndarray,
                        done:bool)->None:
        self.counter+=1
        self.replay_buffer.append((obs,action,reward,next_obs,done))

        if len(self.replay_buffer) > self.replay_start_size and self.counter%self.replay_frequent == 0:
            self.batch_Q_approximation(*self.replay_buffer.sample(self.batch_size))
        
        # Synchronize the parameters of the two Q networks every target_update_frequent steps
        if self.counter%self.target_sync_frequent == 0:
            self.sync_target_QNet()
    
    def get_max_Q(self, obs:np.ndarray)->float:
        obs=torch.tensor(obs,dtype=torch.float32,device=self.device)
        Q_values=self.main_QNet(obs)
        return torch.max(Q_values).item()