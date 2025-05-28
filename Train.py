import copy
import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
import copy
import random 
import matplotlib.pyplot as plt

import Agent
import Network

class TrainManager():
    def __init__(self,
                 env:gym.Env,
                 lr:float = 1e-3,
                 epsilon:float = 0.1,
                 mini_epsilon:float = 0.01,
                 explore_decay_rate:float = 0.0001,
                 buffer_capacity:int = 2000,
                 replay_start_size:int = 200,
                 replay_frequent:int = 4,
                 target_sync_frequent:int = 200,
                 batch_size:int = 32,
                 episode_num:int = 1000,
                 eval_iters:int = 10,
                 gamma:float = 0.9,
                 seed:int = 0,
                 device:torch.device = torch.device("cpu"),
                 dqn_type:str = "DQN"
                 ) -> None:
        
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.train_env=env
        self.eval_env = copy.deepcopy(env)
        _ = self.train_env.reset(seed=self.seed) # Set the seed of the environment
        _ = self.eval_env.reset(seed=self.seed+1)

        self.episode_num=episode_num
        self.gamma=gamma
        self.device=device

        obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        action_dim=env.action_space.n
        if dqn_type == "DQN" or dqn_type == "Double_DQN":
            QNet=Network.QNet(obs_dim,action_dim).to(self.device)
        elif dqn_type == "Dueling_DQN":
            QNet=Network.VANet(obs_dim,action_dim).to(self.device)
        optimizer=torch.optim.Adam(QNet.parameters(),lr=lr)

        self.buffer=Agent.ReplayBuffer(capacity=buffer_capacity,device=self.device)
        self.agent=Agent.DQN(QNet=QNet,
                              epsilon = epsilon,
                              mini_epsilon = mini_epsilon,
                              action_dim=action_dim,
                              optimizer=optimizer,
                              replay_buffer=self.buffer,
                              replay_start_size=replay_start_size,
                              replay_frequent=replay_frequent,
                              target_sync_frequent=target_sync_frequent,
                              batch_size=batch_size,
                              epsilon_decay_rate=explore_decay_rate,
                              gamma=gamma,
                              device=self.device,
                              dqn_type=dqn_type)
        
        self.eval_iters = eval_iters
        self.eval_rewards = []
        self.step_max_Qvalues = []
        
        
    def train_episode(self)->None:
        obs,_=self.train_env.reset()
        done=False
        while not done:
            action=self.agent.get_behavior_action(obs)
            next_obs,reward,terminated, truncated,_=self.train_env.step(action)
            done=terminated or truncated
            self.agent.Q_approximation(obs,action,reward,next_obs,done)
            self.step_max_Qvalues.append(self.agent.get_max_Q(obs))
            obs=next_obs
    
    def eval(self)->float:
        temp_eval_rewards = []
        for _ in range(self.eval_iters):
            obs,_=self.eval_env.reset()
            episode_reward=0
            done=False
            while not done:
                action = self.agent.get_target_action(obs)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs
            temp_eval_rewards.append(episode_reward)
        self.eval_rewards.append(temp_eval_rewards)

        return np.mean(temp_eval_rewards)
    
    def train(self)->list:
        with tqdm(total=self.episode_num,desc="Training") as pbar:
            for _ in range(self.episode_num):
                self.train_episode()
                avg_reward=self.eval()
                pbar.set_postfix({"Avg. Test Reward": (avg_reward)})
                pbar.update(1)
        
        return self.step_max_Qvalues


if __name__ == "__main__":
    env=gym.make("CartPole-v0")
    DQN_Manager=TrainManager(env=env,
                               episode_num = 2000,
                               lr = 1e-3,
                               gamma = 0.9,
                               epsilon = 0.3,
                               target_sync_frequent = 200,
                               mini_epsilon = 0.1,
                               explore_decay_rate = 0.0001,
                               seed = 0,
                               device = "cpu",
                               dqn_type = "DQN")
    max_Qvalues_DQN = DQN_Manager.train()
    
    env=gym.make("CartPole-v0")
    Double_DQN_Manager=TrainManager(env=env,
                               episode_num = 2000,
                               lr = 1e-3,
                               gamma = 0.9,
                               epsilon = 0.3,
                               target_sync_frequent = 200,
                               mini_epsilon = 0.1,
                               explore_decay_rate = 0.0001,
                               seed = 0,
                               device = "cpu",
                               dqn_type = "Double_DQN")
    max_Qvalues_Double_DQN = Double_DQN_Manager.train()

    Dueling_DQN_Manager=TrainManager(env=env,
                               episode_num = 2000,
                               lr = 1e-3,
                               gamma = 0.9,
                               epsilon = 0.3,
                               target_sync_frequent = 200,
                               mini_epsilon = 0.1,
                               explore_decay_rate = 0.0001,
                               seed = 0,
                               device = "cpu",
                               dqn_type = "Dueling_DQN")
    max_Qvalues_Dueling_DQN = Dueling_DQN_Manager.train()

    """ Plot the episode rewards"""
    DQN_Manager.eval_rewards = np.array(DQN_Manager.eval_rewards)
    Double_DQN_Manager.eval_rewards = np.array(Double_DQN_Manager.eval_rewards)
    Dueling_DQN_Manager.eval_rewards = np.array(Dueling_DQN_Manager.eval_rewards)
    dqn_means = np.mean(DQN_Manager.eval_rewards,axis=-1)
    dqn_stds = np.std(DQN_Manager.eval_rewards,axis=-1)
    ddqn_means = np.mean(Double_DQN_Manager.eval_rewards,axis=-1)
    ddqn_stds = np.std(Double_DQN_Manager.eval_rewards,axis=-1)
    dueling_dqn_means = np.mean(Dueling_DQN_Manager.eval_rewards,axis=-1)
    dueling_dqn_stds = np.std(Dueling_DQN_Manager.eval_rewards,axis=-1)
    plt.figure()
    plt.plot(dqn_means,label="DQN")
    plt.fill_between(np.arange(len(dqn_means)),dqn_means-dqn_stds,dqn_means+dqn_stds,alpha=0.2,label="DQN")
    plt.plot(ddqn_means,label="Double DQN")
    plt.fill_between(np.arange(len(ddqn_means)),ddqn_means-ddqn_stds,ddqn_means+ddqn_stds,alpha=0.2,label="Double DQN")
    plt.plot(dueling_dqn_means,label="Dueling DQN")
    plt.fill_between(np.arange(len(dueling_dqn_means)),dueling_dqn_means-dueling_dqn_stds,dueling_dqn_means+dueling_dqn_stds,alpha=0.2,label="Dueling DQN")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.show() 
    

    import pandas as pd        
    """ Plot the max Q value over steps. """
    smoothing_window_maxQ = 100000
    plt.figure()
    maxQ_DQN_smoothed = pd.Series(max_Qvalues_DQN).rolling(smoothing_window_maxQ, min_periods=smoothing_window_maxQ).mean() # smoothing_window = 100000
    maxQ_DDQN_smoothed = pd.Series(max_Qvalues_Double_DQN).rolling(smoothing_window_maxQ, min_periods=smoothing_window_maxQ).mean()
    maxQ_Dueling_DQN_smoothed = pd.Series(max_Qvalues_Dueling_DQN).rolling(smoothing_window_maxQ, min_periods=smoothing_window_maxQ).mean()
    plt.plot(maxQ_DQN_smoothed,label="DQN")
    plt.plot(maxQ_DDQN_smoothed,label="Double DQN")
    plt.plot(maxQ_Dueling_DQN_smoothed,label="Dueling DQN")
    plt.xlabel('Step')
    plt.ylabel('Step Max Q Value')
    plt.title("Step Max Q Value Over Steps")
    plt.legend()
    plt.show()