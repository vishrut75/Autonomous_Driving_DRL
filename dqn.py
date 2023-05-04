import random
import numpy as np
from collections import namedtuple, deque
import os
import sys
from Environment import Car_Environment
import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import msvcrt as mm

#from agent import Agent
#from dqn_model import DQN
torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

BufferData = namedtuple('BufferData',('state','state_vel', 'action', 'next_state','nxt_vel', 'reward','done'))

class RainbowDQn(nn.Module):
    def __init__(self,in_dims,n_actions):
        super(RainbowDQn, self).__init__()
        #a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)
        a = Variable(torch.zeros(in_dims)).view(1, -1).size(1)
        self.fc1_adv = nn.Linear(a+28, 256)
        self.fc3_adv = nn.Linear(128, n_actions)
        self.fc2_adv = nn.Linear(256, 128)
        self.fc1_val = nn.Linear(a+28, 512)
        self.fc3_val = nn.Linear(512, 1)
        self.num_actions = n_actions


    def forward(self,x,vel):
        # Flatten
        x = x.view(x.size(0), -1)
        x = torch.cat((x,vel),1)
        x = x.to(torch.float32)
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)
        val = F.relu(self.fc1_val(x))
        val = self.fc3_val(val)
        val = (val - adv.mean(1).unsqueeze(1)).expand(x.size(0), self.num_actions) + adv
        return val


class Agent_DQN():
    def __init__(self):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batchsize = 64 # args.batchsize or 128
        self.gamma = 0.99
        self.epsilon_max = 1.0
        self.epsilon_min = 0.025
        buffersize = 2048
        learning_rate = 1.0e-4
        self.tau = 0.1
        self.net_update_count = 100
        self.step_count = 0
        self.loop_count = 0
        self.n_episodes = 4000 # args.n_episodes or 1000
        self.eps_decay_rate = self.n_episodes/25
        self.carenv = Car_Environment()
        self.n_actions = 6
        in_dims = [1,1,18,32]
        self.policy_net = RainbowDQn(in_dims,self.n_actions).to(self.device)
        self.target_net = RainbowDQn(in_dims,self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_count = 0
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque([],maxlen=buffersize)
        self.replay_count = deque([],maxlen=buffersize)
        self.replay_occur = deque([],maxlen=buffersize)
        self.replay_idx = []
        self.idx_to_del = []
        self.max_priority = 1
        self.project_name = "DQN_v1"
        
        load = True
        if load:
            print("Loaded")
            model_idx = 349//50
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            model_path = "./Models/Actor_DQN_v1%d.pth" %(model_idx)
            tst = torch.load(model_path)
            self.policy_net.load_state_dict(tst)

        self.use_wandb = True
        if self.use_wandb:
            wandb.init(project="Project4-Hope", entity="loser-rl")

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.step_count = 0
        self.loop_count = 0
        self.update_count = 0
        ###########################
        pass

    def manual_input(self):
        char_pressed = mm.getwch()
        if(char_pressed=='a'):
            return 0
        if(char_pressed=='s'):
            return 1
        if(char_pressed=='d'):
            return 2
        if(char_pressed=='f'):
            return 3
        if(char_pressed=='g'):
            return 4
        if(char_pressed=='c'):
            return 5
        return random.randrange(self.n_actions)

    def make_action(self, observation,vel,manual_mode=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if manual_mode:
            return self.manual_input()
        sel = random.random()
        exp_fac = math.exp(-1*self.step_count/self.eps_decay_rate)
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*exp_fac
        if sel > eps or self.step_count%50 == 0:
            policy_out = self.policy_net(observation,vel)
            action_idx = torch.argmax(policy_out,dim=1)[0]
            action = action_idx.detach().item()
            #mn_val = torch.min(policy_out,dim=1)[0]
            #mx_val = torch.max(policy_out,dim=1)[0]
            #if (not test) and (mn_val>0.98*mx_val) and eps<0.1:
                #print("indecisive")
               # print('mn_val',mn_val)
                #print('mx_val',mx_val)
                #return random.randrange(self.n_actions)
        else:
            action = random.randrange(self.n_actions)
        ###########################
        
        return action

    def cvt_act(self,action):
        if action==0:
            return [0.4,0]
        if action==1:
            return [0.6,0.25]
        if action==2:
            return [0.8,0.5]
        if action==3:
            return [0.6,0.75]
        if action==4:
            return [0.4,1.0]
        return [0,0.5]

    def push(self,*args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.replay_occur.append(0)
        self.replay_count.append(self.max_priority)
        self.memory.append(BufferData(*args))


    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        samples = []
        # self.idx_to_del = []
        idxs = random.choices(range(len(self.memory)),weights=self.replay_count,k=self.batchsize)
        # samples = self.memory[self.replay_idx]
        # idxs = random.sample(range(0,len(self.memory)),self.batchsize)
        idxs.sort()
        idxs_val = np.zeros(len(self.memory))
        for i in idxs:
            idxs_val[i] = 1
        j = len(self.memory) - 1
        bs = self.batchsize
        for i in range(bs-1):
            if(idxs[i+1]==idxs[i]):
                while(idxs_val[j]==1):
                    j = random.randrange(len(self.memory))
                idxs[i] = j
                idxs_val[j] = 1
                #idxs[i+1] = random.randrange(len(self.memory))
        # idxs = np.random.choice(len(self.memory),p=self.replay_count,size=self.batchsize,replace=False)
        # samples = self.memory[self.replay_idx]
        # idxs = random.sample(range(0,len(self.memory)),self.batchsize)
        self.replay_idx = idxs
        for idx in idxs:
             samples.append(self.memory[idx])
             cur_count = self.replay_occur[idx]
             # if cur_count==2:
                 # self.idx_to_del.append(idx)
             self.replay_occur[idx]=cur_count+1

        ###########################
        return samples


    def optimize_step(self):
        if(len(self.memory) < 2*self.batchsize):
            return
        print("Training")
        sample = self.replay_buffer()
        batch = BufferData(*zip(*sample))

        next_states = torch.cat(batch.next_state)
        states = torch.cat(batch.state)
        next_vels = torch.cat(batch.nxt_vel)
        vels = torch.cat(batch.state_vel)
        rewards = torch.cat(batch.reward)
        actions = torch.cat(batch.action)
        done = torch.cat(batch.done)
        states.to(self.device)
        next_states.to(self.device)
        vels.to(self.device)
        next_vels.to(self.device)
        
        intm_val = self.policy_net(states,vels)
        state_action_values = intm_val[torch.arange(intm_val.size(0)),actions]
        
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        plcy_act = torch.argmax(self.policy_net(next_states,next_vels),dim=1)[0].detach()
        next_state_intm_val = self.target_net(next_states,next_vels)
        next_state_values = done*next_state_intm_val[torch.arange(next_state_intm_val.size(0)),plcy_act]
        
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        expected_state_action_values = torch.reshape(expected_state_action_values.unsqueeze(1),(1, self.batchsize))[0]
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        intm_val = self.policy_net(states,vels)
        state_action_values = intm_val[torch.arange(intm_val.size(0)),actions]
        prob = (expected_state_action_values-state_action_values).tolist()
        
        self.idx_to_del = []
        max_it = 0
        for i in range(len(prob)):
            cur_val = min(abs(prob[i])**2,10)
            cur_idx = self.replay_idx[i]
            self.replay_count[cur_idx] = cur_val
            max_it = max(cur_val,max_it)
            if cur_val<0.0001 or self.replay_occur[cur_idx]>=3:
                self.idx_to_del.append(cur_idx)
        self.max_priority = max(0.9*self.max_priority,max_it)+0.0001
        print('max_it',max_it)
        
        self.idx_to_del.sort(reverse=True)
        # print(len(self.memory))
        for idx in self.idx_to_del:
            # print(idx)
            del self.memory[idx]
            del self.replay_count[idx]
            del self.replay_occur[idx]
            
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        #episode_time = []
        #episode_reward = []
        ep_r = 0
        ep_t = 0
        max_ep_r = 0
        ne = 349
        self.step_count = ne
        while ne<=self.n_episodes:
            ne+=1
            manual_mode = False
            self.step_count+=1
            state, lidar_vel = self.carenv.reset()
            state = torch.from_numpy(np.transpose(np.asarray(state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
            state = state.to(self.device)
            lidar_vel = torch.tensor([lidar_vel], dtype=torch.float64,device =self.device )
            done = False
            queue_size = 5
            queue = deque([],maxlen=queue_size)
            tdn_reward = 0
            cur_ep_r = 0
            negative_cnt = 0
            #if ne%50==0:
            #    manual_mode = True
            #    print("Manual Mode Started")
            #    print("Use a s d f g to steer")
            while not done:
                a = self.make_action(state,lidar_vel,manual_mode)
                acti = self.cvt_act(a)
                next_state, r, done, nxt_lidar_vel, stale = self.carenv.step(acti)
                #next_state, r, done, _, _ = self.env.step(a)
                cur_ep_r += r
                tdn_reward += r*(self.gamma**len(queue))
                a_tensor = torch.tensor([a], dtype=torch.int64, device=self.device)
                queue.append((state,lidar_vel,a_tensor,r))
                
                if stale:
                    negative_cnt += 1
                else:
                    negative_cnt = 0

                if(negative_cnt>20 or cur_ep_r < -20):
                    done = True


                if done:
                    n_state = state
                    n_vel = lidar_vel
                else:
                    n_state = torch.from_numpy(np.transpose(np.asarray(next_state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
                    n_vel = torch.tensor([nxt_lidar_vel], dtype=torch.float64,device =self.device )
                n_state = n_state.to(self.device)
                done_tensor = torch.tensor([1-done], dtype=torch.int64,device=self.device)
                # r_tensor = torch.tensor([r], dtype=torch.float32, device=self.device)
                # self.push(state,a_tensor,n_state,r_tensor,done_tensor)
                if len(queue)==queue_size:
                    state_tensor,vels_tensor,a_old_tensor,r_old = queue.popleft()
                    r_tensor = torch.tensor([tdn_reward], dtype=torch.float32, device=self.device)
                    self.push(state_tensor,vels_tensor,a_old_tensor,n_state,n_vel,r_tensor,done_tensor)
                    tdn_reward = (tdn_reward - r_old)/self.gamma
                state = n_state
                lidar_vel = n_vel
                self.optimize_step()
                self.loop_count+=1
                ep_t += 1
                if self.loop_count%2500==0:
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)
                    self.loop_count=0
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            max_ep_r = max(max_ep_r,cur_ep_r)
            ep_r += cur_ep_r
            while len(queue)>0:
                state_tensor,vels_tensor,a_old_tensor,r_old = queue.popleft()
                tdn_reward -= r_old
                tdn_reward /= self.gamma
                r_tensor = torch.tensor([tdn_reward], dtype=torch.float32, device=self.device)
                self.push(state_tensor,vels_tensor,a_old_tensor,n_state,n_vel,r_tensor,done_tensor)
            print("Episode %d Completed at %d steps with %d reward" % (ne,ep_t,ep_r))
            #if ne%100==0:
                #print(ep_r/20)
            #episode_reward.append(ep_r)
            exp_fac = math.exp(-1*self.step_count/self.eps_decay_rate)
            epsilon_val = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*exp_fac
            if(self.use_wandb):
                wandb.log({"episode":ne,"epsilon":epsilon_val,"episode_reward": ep_r,"max_reward": max_ep_r})
            ep_r = 0
            # if ne%1000==0:
                # self.target_net.load_state_dict(self.policy_net.state_dict())
            if ne%50==49:
                model_save_path = "./Models/Actor_DQN_v1%d.pth" %(ne//50)
                print('Saved')
                try:
                    torch.save(self.target_net.state_dict(),model_save_path)
                except:
                    print("some issue with saving")

            #episode_time.append(ep_t)
            if(self.use_wandb):
                wandb.log({"episode":ne,"episode_time": ep_t})
            ep_t = 0
            

agent = Agent_DQN()
agent.train()