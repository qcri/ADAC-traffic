import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Used for Atari
class MH_Conv_Q(nn.Module):
    def __init__(self, frames, num_actions,
                 num_heads: int,
                 transform_strategy: str = None,
                 **kwargs):

        super(MH_Conv_Q, self).__init__()
        self.num_actions = num_actions
        self.num_heads = num_heads
        self._transform_strategy = transform_strategy
        self._kwargs = kwargs

        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.q1 = nn.Linear(3136, 512)
        self.q2 = nn.Linear(512, num_actions * num_heads)

    #def set_transform_matrix(self, matrix):
    #    self._kwargs['transform_matrix'] = matrix

    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))
        c = F.relu(self.c3(c))

        q = F.relu(self.q1(c.reshape(-1, 3136)))
        q = self.q2(q)
        unordered_q_heads = torch.reshape(q, [-1, self.num_actions, self.num_heads])
        q_heads, q_values = combine_q_functions(
            unordered_q_heads, self._transform_strategy, **self._kwargs)
        return (q_heads, unordered_q_heads, q_values)


# Used for Box2D / Toy problems
class MH_FC_Q(nn.Module):
    
    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 num_heads: int,
                 transform_strategy: str = None,
                 **kwargs):
        super(MH_FC_Q, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_heads = num_heads
        self._transform_strategy = transform_strategy
        self._kwargs = kwargs

        self.q1 = nn.Linear(state_dim, 256)
        self.q2 = nn.Linear(256, 256)
        self.q3 = nn.Linear(256, num_actions * num_heads)

    #def set_transform_matrix(self, matrix):
    #    self._kwargs['transform_matrix'] = matrix

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        q = self.q3(q)
        unordered_q_heads = torch.reshape(q, [-1, self.num_actions, self.num_heads])
        q_heads, q_values = combine_q_functions(
            unordered_q_heads, self._transform_strategy, **self._kwargs)
        return (q_heads, unordered_q_heads, q_values)

    
def combine_q_functions(q_functions, transform_strategy, **kwargs):
    """Utility function for combining multiple Q functions.
    Args:
    q_functions: Multiple Q-functions concatenated.
    transform_strategy: str, Possible options include (1) 'IDENTITY' for no
      transformation (2) 'STOCHASTIC' for random convex combination.
    **kwargs: Arbitrary keyword arguments. Used for passing `transform_matrix`,
      the matrix for transforming the Q-values if the passed
      `transform_strategy` is `STOCHASTIC`.
    Returns:
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
    """
    # Create q_values before reordering the heads for training
    q_values = torch.mean(q_functions, dim=-1)

    if transform_strategy == 'STOCHASTIC':
        left_stochastic_matrix = kwargs.get('transform_matrix')
        if left_stochastic_matrix is None:
            raise ValueError('None value provided for stochastic matrix')
        q_functions = torch.matmul(
            q_functions, left_stochastic_matrix) # [bsz, actions, heads] * [heads, convex_combos]
    elif transform_strategy == 'IDENTITY':
        pass #tf.logging.info('Identity transformation Q-function heads')
    else:
        raise ValueError(
            '{} is not a valid reordering strategy'.format(transform_strategy))
    return q_functions, q_values


def random_stochastic_matrix(dim, num_cols=None, dtype=torch.float32):
    """Generates a random left stochastic matrix."""
    mat_shape = (dim, dim) if num_cols is None else (dim, num_cols)
    mat = torch.rand(mat_shape, dtype=dtype)
    mat /= torch.norm(mat, p=1, dim=0, keepdim=True)
    return mat


class ensemble_DQN(object):
    def __init__(
        self, 
        is_atari,
        num_actions,
        state_dim,
        num_heads,
        num_convex_combos,
        device,
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
        algo_name='REM',
    ):
        self.device = device
        self.algo_name = algo_name
        self.num_heads = num_heads
        self._num_convex_combinations = num_convex_combos if num_convex_combos > 0 else num_heads
        self._q_heads_transform = None
        if algo_name=='REM':
            self.transform_strategy = 'STOCHASTIC'
            self._q_heads_transform = random_stochastic_matrix(
                self.num_heads, num_cols=self._num_convex_combinations).to(self.device)
        else:
            self.transform_strategy = 'IDENTITY'
        
        kwargs = {}
        if self._q_heads_transform is not None:
            kwargs['transform_matrix'] = self._q_heads_transform

        # Determine network type
        self.Q = MH_Conv_Q(state_dim[0], num_actions, num_heads, self.transform_strategy, **kwargs).to(self.device) if is_atari else MH_FC_Q(state_dim, num_actions, num_heads, self.transform_strategy, **kwargs).to(self.device)
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
        
        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                _, _, mean_values = self.Q(state) #[actions]
                max_action = int(mean_values.argmax())
                return max_action
        else:
            return np.random.randint(self.num_actions)
        
    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()
        
        # New Random convex combination
        #if self.algo_name == 'REM':
        #    matrix = random_stochastic_matrix(
        #        self.num_heads, num_cols=self._num_convex_combinations).to(self.device)
        #    self.Q.set_transform_matrix(matrix)
        #    self.Q_target.set_transform_matrix(matrix)
        
        # Pick a combination at random for REM and use mean value for Ensemble
        #combo = np.random.randint(0, self._num_convex_combinations)
        # Compute the target Q value
        with torch.no_grad():
            q_heads, _, _ = self.Q_target(next_state)
            #q_heads = q_heads[:, :, combo].squeeze(dim=-1) #[bsz*action]
            update, _ = q_heads.max(dim=1, keepdim=True) #[bsz*1*heads]
            update = update.squeeze(dim=1) #[bsz*heads]
            reward = reward #[bsz*1]
            done = done #[bsz*1]
            #print(f'dims: update: {update.shape}, reward: {reward.shape}')
            target_Q = reward + done * self.discount * update #[bsz*heads]
            #print(f'update: {update[:2]}, target q: {target_Q[:2]}')

            
        # Compute the Q value of the current choice
        q_heads, _, mean_values = self.Q(state)
        indices = action.unsqueeze(-1).repeat(1, 1, q_heads.shape[2])
        #print(f'gather indices: {indices.shape}')
        current_Q = q_heads.gather(1, indices).squeeze(dim=1) #[bsz*heads]
        #current_Q = mean_values.gather(1, action) #[bsz*1]
        #print(f'q dim: {current_Q.shape}, current q: {current_Q[:2]}')
            
        # Loss for each head
        q_loss = F.smooth_l1_loss(current_Q, target_Q) #[heads]
        loss = q_loss.mean() #scalar
        
        # Optimize the Q
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        
        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        return q_loss.item()


    def sample_q_values(self, replay_sample):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_sample

        # Compute the target Q value
        with torch.no_grad():
            _, _, mean_values = self.Q(state)
            values = mean_values.gather(1, action)
            (std, mean) = torch.std_mean(values)

        print(f'Sampled q value: {mean.item()} +- {std.item()}')
        return (mean.item(), std.item())

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

