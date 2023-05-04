import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.autograd import Variable
import numpy as np
import msvcrt as mm
import random

class Actor(nn.Module):
    def __init__(self,in_dims,n_actions):
        super(Actor, self).__init__()
        # create network elements
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), # 72,128 -> (144 - 8)/4 + 1 ,  = 16, 30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 6, 13
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1), # 5, 12
            nn.ReLU())
        

        #a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)
        a = Variable(torch.zeros(in_dims)).view(1, -1).size(1)
        self.fc1_adv = nn.Linear(a+28, 256)
        self.fc2_adv = nn.Linear(256, 128)
        self.fc3_adv = nn.Linear(128, n_actions)
        self.num_actions = n_actions


    def forward(self,x,vel):
        # create network
        # output dimension 2 -> Throttle [-1,1] , Steering [-1,1] 
        #x = self.conv(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = torch.cat((x,vel),1)
        x = x.to(torch.float32)
        x = F.relu(self.fc1_adv(x))
        x = F.relu(self.fc2_adv(x))
        #val = F.softmax(self.fc3_adv(x),dim=-1)
        val = torch.tanh(self.fc3_adv(x))

        return val

class SACActor(nn.Module):
    def __init__(self,in_dims,n_actions):
        super(Actor, self).__init__()
        # create network elements
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), # 72,128 -> (144 - 8)/4 + 1 ,  = 16, 30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 6, 13
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1), # 5, 12
            nn.ReLU())
        

        a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)

        self.fc1_adv = nn.Linear(a, 256)
        self.fc2_adv = nn.Linear(256+27, 128)
        self.fc3_adv = nn.Linear(128, n_actions)
        self.fc3_std = nn.Linear(128, n_actions)
        self.num_actions = n_actions


    def forward(self,x,vel):
        # create network
        # output dimension 2 -> Throttle [-1,1] , Steering [-1,1] 
        x = self.conv(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_adv(x))
        x = torch.cat((x,vel),1)
        x = F.relu(self.fc2_adv(x))
        val = F.softmax(self.fc3_adv(x),dim=-1)
        std = F.softmax(self.fc3_std(x),dim=-1)

        return val, std


class Critic(nn.Module):
    def __init__(self,in_dims):
        super(Critic, self).__init__()
        # create network elements
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), # 72,128 -> (144 - 8)/4 + 1 ,  = 16, 30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 6, 13
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1), # 5, 12
            nn.ReLU())
        
        #a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)
        a = Variable(torch.zeros(in_dims)).view(1, -1).size(1)
        self.fc1_val = nn.Linear(a+28, 256)
        self.fc2_val = nn.Linear(256, 128)
        self.fc3_val = nn.Linear(128, 1)
    
    def forward(self,x,vel):
        # create network
        # output dimension 1 -> Value
        #x = self.conv(x)
        #Flatten
        x = x.view(x.size(0), -1)
        x = torch.cat((x,vel),1)
        x = x.to(torch.float32)
        x = F.relu(self.fc1_val(x))
        x = F.relu(self.fc2_val(x))
        x = self.fc3_val(x)
        return x

class AC(nn.Module):
    def __init__(self,in_dims,n_actions):
        super(AC, self).__init__()
        # create network elements
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), # 72,128 -> (144 - 8)/4 + 1 ,  = 16, 30
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 6, 13
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), # 5, 12
            nn.ReLU())
        

        a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)

        self.fc1 = nn.Linear(a, 512)
        self.fc2_adv = nn.Linear(512+26, n_actions)
        self.fc2_val = nn.Linear(512+26, 1)
        self.num_actions = n_actions


    def forward(self,x,vel):
        # create network
        # output dimension 2 -> Throttle [-1,1] , Steering [-1,1] 
        x = self.conv(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.cat((x,vel),1)
        x = x.to(torch.float32)
        act = F.softmax(self.fc2_adv(x),dim=-1)
        val = self.fc2_val(x)

        return act,val


class ActorCriticAgent():
    def __init__(self,in_dims,action_dim,device,action_std_init=0.3):
        self.device = device
        self.network = AC(in_dims,action_dim)
        self.action_dim = action_dim
        #self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        self.action_var = torch.Tensor([0.03,0.03])

    def reset_eps(self):
        self.action_var = torch.Tensor([0.03,0.03])
        print('reset')

    def equal_eps(self,eps_val):
        self.action_var = torch.Tensor(eps_val)

    def set_eps(self):
        self.action_var = torch.Tensor([0.03,0.001])
        print('set')

    def choose_action(self,state,vel,manual_mode=False):
        action_mean, state_val = self.network(state,vel)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action[0][1] = max(action_mean[0][1]-0.15,min(action_mean[0][1]+0.15,action[0][1]))
        if manual_mode:
            char_pressed = mm.getwch()
            action_req = 0.00
            if(char_pressed=='a'):
                action_req=0.00
            if(char_pressed=='s'):
                action_req=0.25
            if(char_pressed=='d'):
                action_req=0.50
            if(char_pressed=='f'):
                action_req=0.75
            if(char_pressed=='g'):
                action_req=1.00
            delta = abs(np.random.normal(0,self.action_var[1],1)[0])
            if(action_req>action_mean[0][1]):
                delta = -1*delta
            if(char_pressed != 'c'):
                action[0][1] = action_req+delta
        #action0_var = self.action_var[0]
        #action0 = random.triangular(action_mean[0][0]-action0_var,action_mean[0][0],action_mean[0][0]+action0_var)
        #action1_var = self.action_var[1]
        #action1 = random.triangular(action_mean[0][1]-action1_var,action_mean[0][1],action_mean[0][1]+action1_var)
        #action = torch.Tensor([[action0,action1]])
        if(action[0][0].item()>1):
            action[0][0]=1.0
        elif(action[0][0].item()<0):
            action[0][0]=0.0

        
        if(action[0][1].item()>1):
            action[0][1]=1.0
        elif(action[0][1].item()<0):
            action[0][1]=0.0
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_action(self,state,vel,action):
        action_mean, state_val = self.network(state,vel)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_val, dist_entropy

    def get_state_dict(self):
        return self.network.state_dict()

    def load_dict(self,state_dict):
        self.network.load_state_dict(state_dict)

class ActorCritic():
    def __init__(self,in_dims,action_dim,device,action_std_init=0.3):
        self.device = device
        self.actor = Actor(in_dims,action_dim)
        self.critic = Critic(in_dims)
        self.action_dim = action_dim
        #self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        self.eps = 0.2
        self.action_var = torch.Tensor([self.eps,self.eps]).to(self.device)

    def equal_eps(self,eps_val):
        self.action_var = torch.Tensor(eps_val)

    def reset_eps(self):
        self.eps = self.eps/1.012
        self.action_var = torch.Tensor([self.eps,self.eps]).to(self.device)
        print('eps',self.eps)

    def choose_action(self,state,vel,manual_mode=False):
        action_mean = self.actor(state,vel) 
        if vel[0][27].item()>0.3 and action_mean[0][1].item()<0:
            action_mean[0][1] = 0.5
            print("assist")
        if vel[0][27].item()<-0.3 and action_mean[0][1].item()>0:
            action_mean[0][1] = -0.5
            print("assist")
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        state_val = self.critic(state,vel)
        #print(state_val)
        #action[0][1] = max(action_mean[0][1]-0.15,min(action_mean[0][1]+0.15,action[0][1]))
        #if manual_mode:
        #    char_pressed = mm.getwch()
        #    action_req = 0.00
        #    if(char_pressed=='a'):
        #        action_req=0.00
        #    if(char_pressed=='s'):
        #        action_req=0.25
        #    if(char_pressed=='d'):
        #        action_req=0.50
        #    if(char_pressed=='f'):
        #        action_req=0.75
        #    if(char_pressed=='g'):
        #        action_req=1.00
        #    delta = abs(np.random.normal(0,self.action_var[1],1)[0])
        #    if(action_req>action_mean[0][1]):
        #        delta = -1*delta
        #    if(char_pressed != 'c'):
        #        action[0][1] = action_req+deltasssss
        #print(vel[0][27].item())

        if(action[0][0].item()>1):
            action[0][0]=1.0
        elif(action[0][0].item()<-1):
            action[0][0]=-1.0

        
        if(action[0][1].item()>1):
            action[0][1]=1.0
        elif(action[0][1].item()<-1):
            action[0][1]=-1.0
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_action(self,state,vel,action):
        action_mean = self.actor(state,vel)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_val = self.critic(state,vel)

        return action_logprobs, state_val, dist_entropy

    def set_action_std(self,std_val):
        # self.action_var = torch.full((self.action_dim,), std_val * std_val).to(self.device)
        self.action_var = std_val.to(self.device)

    def get_state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def load_dict(self,actor_dict,critic_dict):
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)

    def soft_copy(self,network,tau=0.1):
         target_actor_dict, target_critic_dict = self.get_state_dict()
         policy_actor_dict, policy_critic_dict = network.get_state_dict()
         for key in policy_actor_dict:
             target_actor_dict[key] = policy_actor_dict[key]*tau + target_actor_dict[key]*(1-tau)
         for key in policy_critic_dict:
             target_critic_dict[key] = policy_critic_dict[key]*tau + target_critic_dict[key]*(1-tau)
         self.load_dict(target_actor_dict,target_critic_dict)
  
class SAC():
    def __init__(self,in_dims,action_dim,device,action_std_init=0.3):
        self.device = device
        self.actor = SACActor(in_dims,action_dim)
        self.critic = Critic(in_dims)
        self.action_dim = action_dim

    def choose_action(self,state,vel):
        action_mean, action_std = self.actor(state,vel)
        std1 = min(action_std[0][0],0.08)
        std2 = min(action_std[0][1],0.08)
        self.action_var = torch.Tensor([std1,std2])
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        state_val = self.critic(state,vel)
        if(action[0][0].item()>1):
            action[0][0]=1.0
        elif(action[0][0].item()<0):
            action[0][0]=0.0

        
        if(action[0][1].item()>1):
            action[0][1]=1.0
        elif(action[0][1].item()<0):
            action[0][1]=0.0

        
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_action(self,state,vel,action):
        action_mean, action_std = self.actor(state,vel)
        std1 = action_std[0][0]
        std2 = action_std[0][1]
        self.action_var = torch.Tensor([std1,std2])
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_val = self.critic(state,vel)

        return action_logprobs, state_val, dist_entropy

    def get_state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def load_dict(self,actor_dict,critic_dict):
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)

    def soft_copy(self,network,tau=0.1):
         target_actor_dict, target_critic_dict = self.get_state_dict()
         policy_actor_dict, policy_critic_dict = network.get_state_dict()
         for key in policy_actor_dict:
             target_actor_dict[key] = policy_actor_dict[key]*tau + target_actor_dict[key]*(1-tau)
         for key in policy_critic_dict:
             target_critic_dict[key] = policy_critic_dict[key]*tau + target_critic_dict[key]*(1-tau)
         self.load_dict(target_actor_dict,target_critic_dict)
 