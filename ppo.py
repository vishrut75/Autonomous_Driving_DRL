import random
from sre_parse import State
import numpy as np
from collections import namedtuple, deque


#D:\Epic Games\AirSim\Unreal\Environments\AirSimNH\AirSimNH\WindowsNoEditor
#start AirSimNH -ResX=640 -ResY=480 -windowed

import wandb
import math
import torch
import torch.nn as nn
import torch.optim

from Environment import Car_Environment
from Network import ActorCritic
import random

BufferData = namedtuple('BufferData',('state', 'lidar_vel', 'action', 'value', 'logprobs', 'reward','done','returns','advantage'))

class PPO():
    def __init__(self,buffersize = 512,minibatch=256):
        self.carenv = Car_Environment()
        in_dims = [1,1,18,32]
        action_dim = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ActorCritic(in_dims,action_dim,self.device)
        self.target_net = ActorCritic(in_dims,action_dim,self.device)
        self.project_name = "PPO_Hope"
        self.buffersize = buffersize
        self.memory = []
        self.minibatch = minibatch
        # buffersize // minibatch
        self.epoch = 3
        self.eps_clip = 0.2
        lr_actor = 0.5e-4
        lr_critic = 1.0e-4
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_net.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_net.critic.parameters(), 'lr': lr_critic}
                    ])

        load = True
        if load:
            print("Loaded")
            model_idx = 1499//50
            Actor = torch.load('./Models/Actor_PPO_Hope'+str(model_idx)+'.pth')
            Critic = torch.load('./Models/Critic_PPO_Hope'+str(model_idx)+'.pth')
            # self.target_net.load_dict(Actor,Critic)
            self.policy_net.load_dict(Actor,Critic)
            #Critic = torch.load('./Models/Agent_PPO_v8'+str(model_idx)+'.pth')
            #self.policy_net.load_dict(Critic)
        self.target_net.soft_copy(self.policy_net,1.0)

        self.use_wandb = True
        if self.use_wandb:
            wandb.init(project=self.project_name, entity="loser-rl")

    def push(self,*args):
        self.memory.append(BufferData(*args))
        if(len(self.memory)>self.buffersize):
            self.memory.pop(0)

    def sample_batch(self):
        ##if(self.minibatch >= len(self.memory)+1):
        ##    rand_idx = list(range(0,len(self.memory)))
        ##    random.shuffle(rand_idx)
        ##else:
        ##    rand_idx = np.random.randint(len(self.memory),size=self.minibatch)
        samples = []
        ##for idx in rand_idx:
        ##    samples.append(self.memory[idx])
        mem_len = len(self.memory)
        for j in range(mem_len//32):
            samples.append(self.memory[j*32:min(mem_len,(j+1)*32)])
        return samples
        #return [self.memory]

    def gae(self,next_state_value,gamma=0.99,tau=0.95):
        cur_idx = len(self.memory) -  1
        done = 0
        rewards = []
        gae = 0
        sample = self.memory[cur_idx]
        while cur_idx >= 0:
            state_value = sample.value
            reward = sample.reward
            delta = reward + gamma*next_state_value - state_value
            gae = delta + gamma*tau*gae

            rewards.append(torch.tensor([gae+state_value]))

            cur_idx -= 1
            #next_state_value = (1-sample.done)*(reward + gamma*next_state_value)
            next_state_value = state_value
            if(cur_idx >=0 ):
                sample = self.memory[cur_idx]
                done = sample.done
            if(done):
                next_state_value = 0.0
                gae = 0

        rewards.reverse()
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        cur_idx += 1
        new_idx = 0
        while cur_idx < len(self.memory):
            buffersample = self.memory[cur_idx]
            buffersample._replace(returns=rewards[new_idx])
            self.memory[cur_idx] = buffersample
            new_idx += 1
            cur_idx += 1

    def optimize_step(self):
        self.carenv.pause()
        samples = self.sample_batch()
        for _ in range(self.epoch):
            for sample in samples:
                batch = BufferData(*zip(*sample))
                states = torch.cat(batch.state)
                lidar_vels = torch.cat(batch.lidar_vel)
                actions = torch.cat(batch.action)
                old_logprobs = torch.cat(batch.logprobs)
                old_values = torch.cat(batch.value)
                # advantages = torch.cat(batch.advantage)
                returns = torch.cat(batch.returns)

                advantages = returns.detach() - old_values.detach()
                advantages = (advantages - advantages.mean())/(advantages.std() + 1e-7)

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy_net.evaluate_action(states,lidar_vels,actions)
            
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01*dist_entropy
            
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        self.target_net.soft_copy(self.policy_net,0.5)

            
        self.carenv.resume()

    def train_new(self,n_episodes = 2000):
        print('Started')
        epn =1500
        while epn<=n_episodes:
            state, lidar_vel = self.carenv.reset()
            state = torch.from_numpy(np.transpose(np.asarray(state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
            state = state.to(self.device)
            lidar_vel = torch.tensor([lidar_vel], dtype=torch.float64,device =self.device )
            done = 0
            ept = 0
            epr = 0
            negative_cnt = 0
            while not done:
                with torch.no_grad():
                    # Choose action and take action
                    action, logprob, value = self.target_net.choose_action(state,lidar_vel,False)

                nxt_state, reward, done, nxt_lidar_vel, stale = self.carenv.step(action.numpy()[0])
                
                #increment step count
                ept += 1
                epr += reward

                if stale:
                    negative_cnt += 1
                else:
                    negative_cnt = 0

                if(negative_cnt>=30 or epr <= -30):
                    done = 1

                #append to memory buffer
                self.push(state, lidar_vel, action, value, logprob, reward,done,torch.tensor([0.0]),torch.tensor([0.0]))
                state = torch.from_numpy(np.transpose(np.asarray(nxt_state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
                state = state.to(self.device)
                lidar_vel = torch.tensor([nxt_lidar_vel])
                lidar_vel = lidar_vel.to(self.device)
                if(len(self.memory)>=self.minibatch):
                    print('Training')
                    nxt_value = self.policy_net.critic(state,lidar_vel)
                    nxt_value = nxt_value.detach()
                    self.gae(nxt_value*(1-done))
                    self.optimize_step()
                    del self.memory[0:127]
                    self.policy_net.reset_eps()
            

            print("Episode %d Completed at %d steps with %d reward" % (epn,ept,epr))
            if self.use_wandb:
                wandb.log({"episode":epn,"episode_time": ept,"episode_reward": epr})
            if epn%50 == 49:
                self.save_checkpoint('./Models',epn//50)
            epn+=1


    def save_checkpoint(self,path,epn=0):
        print("Saved %d" %(epn))
        actor_dict, critic_dict = self.target_net.get_state_dict()
        torch.save(actor_dict,path+'/Actor_'+self.project_name+str(epn)+'.pth')
        torch.save(critic_dict,path+'/Critic_'+self.project_name+str(epn)+'.pth')

model = PPO();
model.train_new();




