import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)		

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return F.softmax(self.l3(a), dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)


    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
        def __init__(
            self,
            is_atari,
            num_actions,
            state_dim,
            device,
            alpha=1.0, ##verify if 1 is suitable in discrete setting
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            algo_name='TD3BC'
            ):
                self.num_actions = num_actions
                self.state_dim = state_dim
                self.device = device
                self.actor = Actor(state_dim, num_actions).to(device)
                self.actor_target = copy.deepcopy(self.actor)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), **optimizer_parameters)

                self.critic = Critic(state_dim, num_actions).to(device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), **optimizer_parameters)

                self.discount = discount
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                self.alpha = alpha

                # Target update rule
                #self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
                self.target_update_frequency = target_update_frequency
                self.tau = tau

                # Evaluation stat
                self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode

                self.total_it = 0


        def select_action(self, state, eval=False):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().argmax()
            self.action_stat[action] += 1
            return action

        def print_stat(self):
                print(f'After evaluation:\n Action count stat: {self.action_stat}')
                # reset stats
                self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode


        def train(self, replay_buffer, batch_size=256):
                self.total_it += 1

                # Sample replay buffer 
                state, action, next_state, reward, not_done = replay_buffer.sample()

                with torch.no_grad():
                        # Select action according to policy and add clipped noise
                        #noise = (
                        #        torch.randn_like(action) * self.policy_noise
                        #        ).clamp(-self.noise_clip, self.noise_clip)

                        next_action = self.actor_target(next_state).argmax(1, keepdim=True)

                        # Compute the target Q value
                        target_Q = self.critic_target(next_state)
                        target_Q = reward + not_done * self.discount * target_Q.gather(1, next_action).reshape(-1,1)

                # Get current Q estimates
                current_Q = self.critic(state).gather(1, action)

                # Compute critic loss
                critic_loss = F.smooth_l1_loss(current_Q, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Delayed policy updates
                if self.total_it % self.policy_freq == 0:

                        # Compute actor loss
                        pi = self.actor(state)
                        action_tensor = (action.detach() == torch.arange(self.num_actions).reshape(1, self.num_actions).to(self.device)).float()
                        # print('action_tensor shape: ', action_tensor.shape, ' value: ', action_tensor)
                        #Q = self.critic(state, pi)
                        ## mayuresh: changing denominator from mean to max to make the range of q loss from (0, 1], verify!
                        lmbda = self.alpha/current_Q.abs().mean().detach()

                        actor_loss = -lmbda * (self.critic(state)*pi).sum(dim=1).mean() + F.mse_loss(pi, action_tensor) 

                        # Optimize the actor 
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # Update the frozen target models
                        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        def sample_q_values(self, replay_sample):
                # Sample replay buffer
                state, action, next_state, reward, done = replay_sample

                # Compute the target Q value
                with torch.no_grad():
                        q = self.critic(state)
                        values = q.gather(1, action)
                        (std, mean) = torch.std_mean(values)

                print(f'Sampled q value: {mean.item()} +- {std.item()}')
                return (mean.item(), std.item())

        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

                torch.save(self.actor.state_dict(), filename + "_actor")
                torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


        def load(self, filename):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.critic_target = copy.deepcopy(self.critic)

                self.actor.load_state_dict(torch.load(filename + "_actor"))
                self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
                self.actor_target = copy.deepcopy(self.actor)

