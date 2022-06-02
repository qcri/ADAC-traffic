import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		self.q1 = nn.Linear(3136, 512)
		self.q2 = nn.Linear(512, num_actions)

		self.i1 = nn.Linear(3136, 512)
		self.i2 = nn.Linear(512, num_actions)


	def forward(self, state):
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		q = F.relu(self.q1(c.reshape(-1, 3136)))
		i = F.relu(self.i1(c.reshape(-1, 3136)))
		i = self.i2(i)
		return self.q2(q), F.log_softmax(i, dim=1), i


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = self.i3(i)
		return self.q3(q), F.log_softmax(i, dim=1), i


class discrete_BCQ(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps = 0.001,
		algo_name='BCQ',
		sm_threshold = 5,
		mm_threshold = 5
	):
	
		self.device = device

		# Algorithm
		self.no_target = False #Use a separate target Q network
		#if algo_name in ['DQN', 'DDQN']:
		#	BCQ_threhold = 0 #Disabling BCQ filtering of actions
		if algo_name == 'BC':
			BCQ_threhold = 0.99 #Behavior cloning means filter everything but max value
		if algo_name in ['DQN', 'SM-DQN', 'MM-DQN']:
			print('Using single Q network')
			self.no_target = True #No target Q net
		self.algo_name = algo_name

		# Determine network type
		self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = 0 #eval_eps ##Hack: forcing no evaluation randomness
		self.num_actions = num_actions

		# Threshold for "unlikely" actions 
		self.threshold = BCQ_threshold
		self.sm_threshold = sm_threshold #Inverse Tau parameter for softmax
		self.mm_threshold = mm_threshold #Omega parameter for mellowmax

		# Number of training iterations
		self.iterations = 0

		# Evaluation stat
		self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode
        
	def _aggregate_state_qvalues(self, q):
		# Evaluate max or softmax according to algo_name on q-value vector of 'num_actions' length 
		# Return aggregate value as well as probability distribution over actions
		if True:
			sm = nn.Softmax(dim=1)
			if self.algo_name in ['SM-DQN', 'SM-DDQN']: #softmax
				prob = sm(self.sm_threshold * q) # threshold acts as inverse temperature here
				value = q.mul(prob).sum(dim=1)
				return (value, prob)
			elif self.algo_name in ['MM-DQN', 'MM-DDQN']: #mellowmax
				prob = sm(self.mm_threshold * q) # TODO: threshold here should be adaptive and a function of mm_threshold. See [Assadi, Littman]
				## The following value function will tend to Inf when mm_threshold > 1, use logsumexp trick instead
				#value = torch.log(torch.exp(self.mm_threshold * q).sum(dim=1)/self.num_actions)/self.mm_threshold
				value = (torch.logsumexp(self.mm_threshold * q, dim=1) - torch.log(torch.tensor(float(self.num_actions))))/self.mm_threshold
				return (value, prob)
			else: #regular max
				(value, indices) = q.max(1)
				return (value, F.one_hot(indices, num_classes=self.num_actions))


	def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > self.eval_eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				q, imt, i = self.Q(state)
				max_action = int(q.argmax(1))
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				# Use large negative number to mask actions from argmax
				gen_filtered_max = int((imt * q + (1. - imt) * -1e8).argmax(1))
				if max_action != gen_filtered_max:
					pass#print(f'Max actions differ: {max_action} and {gen_filtered_max}, imt={imt}')
				self.action_stat[gen_filtered_max] += 1
				return gen_filtered_max
		else:
			action = np.random.randint(self.num_actions)
			self.action_stat[action] += 1
			return action

	def print_stat(self):
		print(f'After evaluation:\n Action count stat: {self.action_stat}')
		# reset stats
		self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode

	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			if self.threshold > 0: # use generator network to filter out low probable actions
				q, imt, i = self.Q_target(next_state) if self.no_target else self.Q(next_state) #This network should be Q for double-dqn and Q_target for dqn
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

				(value, prob) = self._aggregate_state_qvalues(imt * q + (1 - imt) * -1e8) # batch constrained softmax
				#print(f'NON bcq: returned numbers: {value[:2]}, {prob[:2,:]}')

				# Use large negative number to mask actions from argmax
				#next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
				#print(f'next action: {next_action[:2, :]}, with imt {imt[:2, :]}')
				q, imt, i = self.Q_target(next_state)
				#print(f'NON bcq: update: {q.mul(prob).sum(dim=1).unsqueeze(1)[:2]}')
				#update = q.gather(1, next_action).reshape(-1, 1)
				nonbcq_update = value.unsqueeze(1) if self.no_target else q.mul(prob).sum(dim=1).unsqueeze(1)
				#print(f'next action: {next_action[:2, :]}, on q {q[:2,:]}, update: {update[:2]}')
				target_Q = reward + done * self.discount * nonbcq_update
				#print(f'updated target: {update[:2]}, nonbcq: {nonbcq_update[:2]}')

			else: # no generator business
				if self.no_target: # use single network i.e. DQN or SM-DQN or MM-DQN
					q, imt, i = self.Q_target(next_state)
					(value, prob) = self._aggregate_state_qvalues(q)
					target_Q = reward + done * self.discount * value.unsqueeze(1)
				else: # use separate target network i.e. DDQN or SM-DDQN or MM-DDQN
					q, imt, i = self.Q(next_state)
					(value, prob) = self._aggregate_state_qvalues(q)
					q, imt, i = self.Q_target(next_state)
					if self.algo_name == 'MM-DDQN': # prob distribution is not over q-values in this case, see SM2 paper
						#update = (self.mm_threshold*q).exp().mul(prob).sum(dim=1).log()/self.mm_threshold
						update = q.mul(prob).sum(dim=1).unsqueeze(1) #TODO: equivalent to softmax technically, but temp. has to be made adaptive
						target_Q = reward + done * self.discount * update
					else:
						target_Q = reward + done * self.discount * q.mul(prob).sum(dim=1).unsqueeze(1)


		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()
		return((q_loss.item(), i_loss.item()))

	def sample_q_values(self, replay_sample):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_sample

		# Compute the target Q value
		with torch.no_grad():
			q, imt, i = self.Q(state)
			values = q.gather(1, action)
			(std, mean) = torch.std_mean(values)

		print(f'Sampled q value: {mean.item()} +- {std.item()}')
		return (mean.item(), std.item())

	def get_q_model(self):
		return self.Q ## required for dac-mdp

	def save(self, filename):
		torch.save(self.Q.state_dict(), filename + "_Q")
		torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

	def load(self, filename):
		self.Q.load_state_dict(torch.load(filename + "_Q"))
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))

	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())