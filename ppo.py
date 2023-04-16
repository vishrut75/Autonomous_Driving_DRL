import random
from sre_parse import State
import numpy as np
from collections import namedtuple, deque


#D:\Epic Games\AirSim\Unreal\Environments\CityEnv\CityEnviron\CityEnviron\WindowsNoEditor>start CityEnviron -ResX=640 -ResY=480 -windowed

# import wandb
import math
import torch
import torch.nn as nn
import torch.optim

from Environment import Car_Environment
from Network import ActorCritic
import random

BufferData = namedtuple('BufferData',('state', 'vel', 'action', 'value', 'logprobs', 'reward','done','returns','advantage'))

class PPO():
    def __init__(self,buffersize = 2048,minibatch=64):
        self.carenv = Car_Environment()
        in_dims = [1,4,144,256]
        action_dim = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_net = ActorCritic(in_dims,action_dim,self.device)
        self.policy_net = ActorCritic(in_dims,action_dim,self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_name = "test_v1"
        # wandb.init(project=self.project_name, entity="loser-rl")
        self.buffersize = buffersize
        self.memory = []
        self.minibatch = minibatch
        self.epoch = 3
        self.eps_clip = 0.2
        lr_actor = 0.001
        lr_critic = 0.001
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_net.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_net.critic.parameters(), 'lr': lr_critic}
                    ])

    def push(self,*args):
        self.memory.append(BufferData(*args))
        if(len(self.memory)>self.buffersize):
            self.pop(0)

    def sample_batch(self):
        if(self.minibatch >= len(self.memory)+1):
            return self.memory
        rand_idx = np.random.randint(len(self.memory),size=self.minibatch)
        samples = []
        for idx in rand_idx:
            samples.append(self.memory[idx])
        return samples

    def return_and_advantage(self,next_state_value,gamma=0.99):
        cur_idx = len(self.memory) -  1
        done = 0
        rewards = []
        # state_value_list = []
        advantages = []
        sample = self.memory[cur_idx]
        while cur_idx >= 0 and done==0:
            state_value = sample.value
            reward = sample.reward
            obs_state_value = reward + gamma*next_state_value

            rewards.append(torch.tensor([obs_state_value]))
            advantages.append(torch.tensor([obs_state_value - state_value]))

            cur_idx -= 1
            next_state_value = obs_state_value
            if(cur_idx >=0 ):
                sample = self.memory[cur_idx]
                done = sample.done

        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # advantages = np.array(rewards) - np.array(state_value_list)

        rewards.reverse()
        advantages.reverse()

        cur_idx += 1
        new_idx = 0
        while cur_idx < len(self.memory):
            buffersample = self.memory[cur_idx]
            buffersample._replace(returns=rewards[new_idx])
            buffersample._replace(advantage=advantages[new_idx])
            self.memory[cur_idx] = buffersample
            new_idx += 1
            cur_idx += 1

    def optimize_step(self):
        self.carenv.pause()
        for _ in range(self.epoch):
            sample = self.sample_batch()
            batch = BufferData(*zip(*sample))
            states = torch.cat(batch.state)
            vels = torch.cat(batch.vel)
            actions = torch.cat(batch.action)
            old_logprobs = torch.cat(batch.logprobs)
            advantages = torch.cat(batch.advantage)
            returns = torch.cat(batch.returns)

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_net.evaluate_action(states,vels,actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.target_net.soft_copy(self.policy_net,1)
        self.carenv.resume()


    def train(self,n_episodes = 50):

        for epn in range(n_episodes):
            state, vel = self.carenv.reset()
            state = torch.from_numpy(np.transpose(np.asarray(state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
            state = state.to(self.device)
            vel = torch.tensor([vel])
            vel = vel.to(self.device)
            done = 0
            print("Episode %d Started" % (epn))
            ept = 0
            epr = 0
            while not done:
                # Choose action and take action
                action, logprob, value = self.target_net.choose_action(state,vel)
                nxt_state, reward, done, nxt_vel = self.carenv.step(action.numpy()[0])

                #increment step count
                ept += 1
                epr += reward

                #append to memory buffer
                self.push(state, vel, action, value, logprob, reward,done,torch.tensor([0.0]),torch.tensor([0.0]))
                state = torch.from_numpy(np.transpose(np.asarray(nxt_state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
                state = state.to(self.device)
                vel = torch.tensor([nxt_vel])
                vel = vel.to(self.device)
            print("Episode %d Completed at %d steps with %d reward" % (epn,ept,epr))
            self.return_and_advantage(0.0)
            self.optimize_step()
            if epn%10 == 99:
                # wandb.log({"episode":epn,"episode_time": ept,"episode_reward": epr})
                self.save_checkpoint('./Models')


    def save_checkpoint(self,path):
        actor_dict, critic_dict = self.target_net.get_state_dict()
        torch.save(actor_dict,path+'/Actor_'+self.project_name+'.pth')
        torch.save(critic_dict,path+'/Critic_'+self.project_name+'.pth')

model = PPO();
model.train();



