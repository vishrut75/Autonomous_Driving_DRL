import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self,in_dims,n_actions):
        super(Actor, self).__init__()
        # create network elements
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 144,256 -> (144 - 8)/4 + 1 ,  = 35, 63
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=4), # 8, 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), # 7, 13
            nn.ReLU())
        

        a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)

        self.fc1_adv = nn.Linear(a+2, 256)
        self.fc2_adv = nn.Linear(256, 128)
        self.fc3_adv = nn.Linear(128, n_actions)
        self.num_actions = n_actions


    def forward(self,x,vel):
        # create network
        # output dimension 2 -> Throttle [-1,1] , Steering [-1,1] 
        x = self.conv(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = torch.cat((x,vel),1)
        x = F.relu(self.fc1_adv(x))
        x = F.relu(self.fc2_adv(x))
        val = F.softmax(self.fc3_adv(x),dim=-1)
        #val = torch.tanh(self.fc3_adv(x))

        return val

class Critic(nn.Module):
    def __init__(self,in_dims):
        super(Critic, self).__init__()
        # create network elements
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 144,256 -> (144 - 8)/4 + 1 ,  = 35, 63
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=4), # 8, 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), # 7, 13
            nn.ReLU())
        
        a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)
        self.fc1_val = nn.Linear(a+2, 512)
        self.fc2_val = nn.Linear(512, 1)
    
    def forward(self,x,vel):
        # create network
        # output dimension 1 -> Value
        x = self.conv(x)
        #Flatten
        x = x.view(x.size(0), -1)
        x = torch.cat((x,vel),1)
        x = F.relu(self.fc1_val(x))
        x = self.fc2_val(x)
        return x

class ActorCritic():
    def __init__(self,in_dims,action_dim,device,action_std_init=0.3):
        self.device = device
        self.actor = Actor(in_dims,action_dim)
        self.critic = Critic(in_dims)
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

    def choose_action(self,state,vel):
        action_mean = self.actor(state,vel)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state,vel)
        if(action[0][0].item()>1):
            action[0][0]=1.0
        elif(action[0][0].item()<0):
            action[0][0]=0.0

        
        if(action[0][1].item()>1):
            action[0][1]=1.0
        elif(action[0][1].item()<0):
            action[0][1]=0.0

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
        self.action_var = torch.full((self.action_dim,), std_val * std_val).to(self.device)

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
         