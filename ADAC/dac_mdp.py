import numpy as np
import pandas as pd
import xxhash
import torch
import hiive.mdptoolbox.mdp as mbox
import random
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from annoy import AnnoyIndex
from tqdm import tqdm

## ANNOY index for kNN search
class ANN(object):    
    def __init__(self, dim, measure='euclidean'):
        self.dim = dim
        self.built = False
        self.annoy = AnnoyIndex(dim, measure)
    
    def index(self, intid, vector):
        assert vector.shape[0] == self.dim
        self.annoy.add_item(intid, vector)
        
    def build(self):
        if not self.built:
            self.annoy.build(5)
        self.built = True
    
    ## alpha is the max distance threshold
    def topn_nn(self, vector, n, alpha=0):
        if not self.built:
            self.build()
        indexes, distances = self.annoy.get_nns_by_vector(vector, n, include_distances=True)
        if alpha > 0:
            idx = next((i for i,v in enumerate(distances) if v > alpha), len(distances))
            idx = max(2, idx) # fetching at least 2 neighbors
            indexes = indexes[0:idx]
            distances = distances[0:idx]
        return indexes, distances
    
    def topd_nn(self, vector, d):
        found_all = False
        n = 10
        cnt = 0
        ## fetch topn results in multiples of 10 until the distances are within d
        while not found_all:
            indexes, distances = self.topn_nn(vector, (1 + cnt) * n)
            #tqdm.write(f'Nearest distances found: {distances} and neighbors: {indexes}')
            if distances[-1] > d or len(distances) < n:
                found_all = True
                idx = next((i for i,v in enumerate(distances) if v > d), len(distances))
                idx = max(2, idx) # fetching at least 2 neighbors
                indexes = indexes[0:idx]
                distances = distances[0:idx]
                #tqdm.write(f'Returning distances: {distances}, neighbors: {indexes}')
                break
            cnt += 1
        return indexes, distances

## UTIL function: Convert a state array to hash digest id
def state_to_id(state):
    assert state is not None
    if isinstance(state, tuple) or isinstance(state, list):
        state = np.array(state)
    return xxhash.xxh3_64(state).hexdigest()


class dac_policy(object):

    def __init__(self,
                 num_actions,
                 state_dim,
                 replay_df,
                 core_states,
                 q_model,
                 device,
                 nn_indexes,
                 k,
                 cost,
                 k_pi,
                 gamma = 0.96,
                 epsilon = 0.01,
                 nn_mode = 'distance',
                 diameter = 1
                ):
        self.num_actions = num_actions
        self.state_dim = state_dim
        
        self.replay_df = replay_df
        self.core_states = core_states

        self.k = k
        self.cost = cost
        self.k_pi = k_pi
        self.gamma = gamma
        self.epsilon = epsilon
        self.nn_mode = nn_mode
        self.diameter = diameter
        assert self.diameter > 0
        self.rmax = self.replay_df['reward'].max()
        self.alpha = 0.8 # The maximum distance threshold to avoid far-off neighbors

        self.q_model = q_model
        self.device = device
        self.nn_indexes = nn_indexes
        self.activation = {}
        if self.q_model: self.q_model.q2.register_forward_hook(self._get_activation('q2'))

        self.transitions, self.rewards = self._build_finite_mdp()

        self.action_value_stat = []
        self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode
        self.known_policy_stat = 0

    
    ## input is nn distances d, convert to sims = 1/(d+\small_number) first, then normalize the sims 
    def _distance_to_alpha(self, indexes, distances, next_state):
        if len(distances) == 0: return np.zeros(0)
        ## weighted sum version
        #delta_d = np.float64(1e-15)
        #sims = np.array([np.float64(1.0) / (d + delta_d) for d in distances])
        #return sims / sims.sum()
        
        ## max reward version
        '''
        alphas = np.zeros(len(distances), dtype=np.float64) 
        max_reward = 0
        alpha_index = 0
        for i, idx in enumerate(indexes):
            if next_state == -1 or next_state == state_to_id(self.replay_df['next_state'][idx]):
                reward = self.replay_df['reward'][idx]
                if reward > max_reward:
                    max_reward = reward
                    alpha_index = i
        alphas[alpha_index] = np.float64(1.0)
        return alphas
        '''
        ## 1/k version
        return np.array([np.float64(1.0)/len(distances) for d in distances])
    
    ## build dac_mdp on core states
    def _build_finite_mdp(self):
        assert self.replay_df is not None
        assert 'core_index' in self.replay_df
        
        stat_distances = []
        stat_max_distance = []
        stat_rewards = []
        stat_rewards_mean = []
        stat_rewards_std = []
        
        #transitions = np.zeros((self.num_actions, len(self.core_states), len(self.core_states)))
        rewards = np.zeros((len(self.core_states), self.num_actions), dtype=np.float64)
        
        #transitions = self.num_actions * [csr_matrix((len(self.core_states), len(self.core_states)), dtype=np.float32)]
        transitions = []
        [transitions.append(lil_matrix(np.zeros((len(self.core_states), len(self.core_states)), dtype=np.float64))) for i in range(self.num_actions)]
        #rewards = self.num_actions * [csr_matrix((len(self.core_states), 1), dtype=np.float64)]
        
        for index, row in tqdm(self.replay_df.iterrows()):
            #action = row['action']
            if row['core_index'] == -1: continue # Ignoring non-core states
            state_rep = self._get_state_rep(row['state'])
            #print(f"On state: {index}, neighbors are: {nn_indexes} with distances: {nn_distances}")
            ## ignore self
            #try:
            #    self_index = nn_indexes.index(index)
            #    del nn_indexes[self_index]
            #    del nn_distances[self_index]
            #except ValueError:
            #    pass
            #print(f"After removing self: {index}, neighbors are: {nn_indexes} with distances: {nn_distances}")
            state_core_index = row['core_index']

            for action in range(self.num_actions):

                if rewards[state_core_index, action] != 0: ## handled before
                    continue

                nn_indexes, nn_distances = self.nn_indexes[action].topn_nn(state_rep, self.k, self.alpha * self.diameter) if self.nn_mode=='number' else self.nn_indexes[action].topd_nn(state_rep, self.k)
                stat_distances.extend(nn_distances) # for statistical purpose
                if len(nn_distances) > 0 and max(nn_distances) < self.alpha * self.diameter:
                    stat_max_distance.append(max(nn_distances))

                transitions[action][state_core_index, :] = np.zeros(len(self.core_states), dtype=np.float64) # initialize row
                #next_state_index = state_to_id(row['next_state'])
                #next_state_core_index = self.core_states.get(next_state_index)
                #transitions[action][state_core_index, next_state_core_index] = np.float(1.0) #HACK: always transitioning to experienced next state, not caring about neighbors
                ## alphas
                nn_alphas = self._distance_to_alpha(nn_indexes, nn_distances, state_to_id(row['next_state']))
                #print(f"Alphas = {nn_alphas}")
                ## max reward
                max_r = self.replay_df['reward'][nn_indexes].max()
                for idx, alpha, d in zip(nn_indexes, nn_alphas, nn_distances):
                    next_state = state_to_id(self.replay_df['next_state'][idx])
                    next_state_core_index = self.core_states.get(next_state)
                    #if next_state_index == next_state: ## satisfies transition requirement
                        #transitions[action, state_core_index, next_state_core_index] += alpha
                    transitions[action][state_core_index, next_state_core_index] += alpha
                    if transitions[action][state_core_index, next_state_core_index] > 1.0:
                        tqdm.write(f'Incosistent row: {action}, {state_core_index} for which value: {transitions[action][state_core_index, next_state_core_index]}, df index: {index}')
                    r_i = self.replay_df['reward'][idx]
                    reward = r_i
                    if self.cost >= 0: # distance factor from DAC-MDP paper
                        reward -= self.cost * d / self.diameter
                    elif self.cost >= -1: # max reward factor
                        reward = reward - max_r * d / self.diameter # normalizing distance before penalizing
                    stat_rewards.append(reward)
                    rewards[state_core_index, action] += alpha * (reward)
                    #rewards[action][state_core_index] += alpha * (r_i - self.cost * d)
                    
        for action in range(self.num_actions):
            tqdm.write(f"transitions on action {action}: {transitions[action].getnnz()}")
        #print(f"Finalized rewards: {rewards}")
                
        ## ensure each row sums to 1 in the transition matrix by putting additional wt on diagonal
        
        changeset = []
        for a, a_mat in enumerate(transitions):
            for idx, s_row in enumerate(a_mat):
                total = s_row.sum()
                if total != np.float64(1.0):
                    #tqdm.write(f"In {a}, {idx}, Total not 1: {total}, diagonal: {transitions[a][idx, :]}")
                    #transitions[a, idx, idx] = 1.0 - total
                    changeset.append((a, idx, idx, (np.float64(1.0) - total)))
                    #transitions[a][(idx, idx)] += (np.float64(1.0) - total)
                    #print(f"Changed diagonal to: {transitions[a][idx, idx]}")
        for (a, row, col, val) in tqdm(changeset):
            transitions[a][(row, col)] += val
        
            
        ## For sanity check, this should NOT be needed
        #for a, a_mat in enumerate(transitions):
        #    for idx, s_row in enumerate(a_mat):
        #        total = s_row.sum()
        #        if total != np.float64(1.0):
        #            tqdm.write(f"In {a}, {idx}, Total not 1: {total}")

        ## transitions to csr
        #print(f"transitions = {transitions} and rewards = {rewards}")
        
        ## stats on reward matrix rows
        for idx, r_row in enumerate(rewards):
            stat_rewards_mean.append(np.mean(rewards[idx]))
            stat_rewards_std.append(np.std(rewards[idx]))

        
        ## print statistics
        #print(f'Rewards: {rewards}, Transitions: {transitions[0]}')
        tqdm.write(f'After building MDP:\n NN distance stats: {stats.describe(stat_distances)}, \n max distance: {stats.describe(stat_max_distance)}, \n NN reward stats: {np.mean(stat_rewards_mean)} +- {np.mean(stat_rewards_std)} and {stats.describe(stat_rewards)}')
        
        return [mat.tocsr() for mat in tqdm(transitions)], rewards
        
    def _get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def _get_state_rep(self, state):
        if self.q_model:
          with torch.no_grad():
            self.q_model(torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(dim=0))
            return self.activation['q2'].squeeze()
        else:
          return np.array(state)
        
    def get_transitions(self):
        return self.transitions
    
    def get_rewards(self):
        return self.rewards
                            
    def solve_mdp(self):
        vi = mbox.ValueIteration(self.transitions, self.rewards, self.gamma, self.epsilon, max_iter=100, skip_check=True)
        vi.run()
        self.policy = vi.policy
        self.value = vi.V
        
        #self.policy = None
        #self.value = np.zeros(len(self.core_states))
        #print(f'Ran VI: policy = {self.policy}, value function = {vi.V}, iter = {vi.iter}, time taken = {vi.time}')
        tqdm.write(f'VI iter = {vi.iter}, time taken = {vi.time}')

    
    def select_action(self, state, eval=False):
        ## lookup in core states
        state_id = state_to_id(state)
        if state_id in self.core_states:
            core_idx = self.core_states.get(state_id)
            action = self.policy[core_idx]
            #print(f'Known actions with values: {self.value[core_idx]}')
            self.action_value_stat.append(self.value[core_idx])
            self.known_policy_stat += 1
        else: # evaluation on non-core state
            state_rep = self._get_state_rep(state)
            action_value = []
            max_distance = 0.0 # for stats only
            for action in range(self.num_actions):
                nn_indexes, nn_distances = self.nn_indexes[action].topn_nn(state_rep, self.k_pi, self.alpha * self.diameter) if self.nn_mode=='number' else self.nn_indexes[action].topd_nn(state_rep, self.k_pi)
                nn_alphas = self._distance_to_alpha(nn_indexes, nn_distances, -1)
                ## max reward
                max_r = self.replay_df['reward'][nn_indexes].max()
                max_distance = max(max_distance, max(nn_distances)) # for stats only
                a_val = 0.0
                for idx, alpha, d in zip(nn_indexes, nn_alphas, nn_distances):
                    state_core_index = self.replay_df['core_index'][idx]
                    next_state_index = state_to_id(self.replay_df['next_state'][idx])
                    next_state_core_index = self.core_states.get(next_state_index)
                    r_i = self.replay_df['reward'][idx]
                    reward = r_i #self.rewards[state_core_index, action]
                    if self.cost >= 0: # distance penalty from DAC-MDP paper 
                        reward -= self.cost * d / self.diameter
                    elif self.cost >= -1: # max reward factor: Adaptive DAC
                        reward = reward - max_r * d / self.diameter  # normalizing distance before penalizing
                    
                    ## Q-value for next state s, action a: R(s, a) + \gamma \sum_s' T(s, a, s') V(s') 
                    max_next_q_value = 0.
                    for a in range(self.num_actions):
                        srow = coo_matrix(self.transitions[a][next_state_core_index]) # coo is faster to iterate, hence convert
                        q_value = self.rewards[next_state_core_index, a]
                        for _, nidx, T in zip(srow.row, srow.col, srow.data): #iterating coo matrix (row)
                            q_value += self.gamma * T * self.value[nidx]
                        #tqdm.write(f'In action {a}, srow {srow}, q_value {q_value}')
                        if q_value > max_next_q_value:
                            max_next_q_value = q_value
                    
                    a_val += alpha * (reward + self.gamma * max_next_q_value) #self.value[next_state_core_index])
                #print(f'On non core state {state}, action {action}: nn indexes={nn_indexes}, distances={nn_distances}, alphas={nn_alphas}, value={a_val}')
                action_value.append(a_val)
            #print(f'Considered actions with values: {action_value}')
            action = action_value.index(max(action_value))
            if max_distance <= self.alpha * self.diameter: ## only for stats
                self.known_policy_stat += 1 ## nearby neighbors found
            self.action_value_stat.append(action_value[action])
        
        self.action_stat[action] += 1
        
        return action
    
    ## to be called at the end of each eval episode
    def print_stat(self):
        tqdm.write(f'After evaluating policy:\n value stats: {stats.describe(np.array(self.value))}, \n Core state matches: {self.known_policy_stat}')
        tqdm.write(f'Action value stats: {stats.describe(np.array(self.action_value_stat))}')
        tqdm.write(f'Action count stat: {self.action_stat}')
        # reset stats
        self.action_value_stat = []
        self.action_stat = [0] * self.num_actions # store the number of occurrences of each action in an eval episode
        self.known_policy_stat = 0

        
class dac_builder(object):
    
    def __init__(self,
                 num_actions,
                 state_dim,
                 buffer,
                 q_model,
                 device,
                 num_configs = 1, # MDP paper provides options 1, 6, 20.
                 nn_mode = 'number', # 'distance' or 'number' or numeric value passing distance diameter
                 diameter = -1,
                 gamma = 0.96,
                 epsilon = 0.01
                ):
        self.num_actions = num_actions
        #print(f'Called with {state_dim} states and {num_actions} actions')
        self.state_dim = state_dim
        self.replay_df = self._build_replay_df(buffer)

        self.core_states = self._detect_core_states()
        self._add_core_state_column()
        
        self.device = device

        self.q_model = None #q_model ##FIXME: Set None only when no_state_rep parameter is on
        self.activation = {}
        if self.q_model: self.q_model.q2.register_forward_hook(self._get_activation('q2'))
        self.ann_indexes = self._build_ann_indexes()

        if diameter <= 0:
            diameter = self._get_core_diameter() ## FIXME: very expensive operation

        self.solvers = []
        if num_configs == 1:
            k = 5 if nn_mode=='number' else 0.02 * diameter
            cost = -0.5
            k_pi = 5 if nn_mode=='number' else 0.04 * diameter
            self.solvers.append(
                dac_policy(
                    num_actions, state_dim, 
                    self.replay_df, self.core_states, 
                    self.q_model, device, self.ann_indexes,
                    k, cost, k_pi,
                    gamma, epsilon, nn_mode, diameter
                )
            )
        elif num_configs == 6:
            k = 5
            for cost in [1, 100, 1e6]:
                for k_pi in [11, 51]:
                    self.solvers.append(
                        dac_policy(
                            num_actions, state_dim, 
                            self.replay_df, self.core_states, 
                            q_model, device, self.ann_indexes,
                            k, cost, k_pi
                        )
                    )
        for solver in tqdm(self.solvers):
            solver.solve_mdp()

    ## Replay buffer
    def _build_replay_df(self, buffer):
        state, action, next_state, reward, not_done = buffer.fetch_all() # returns entire buffer
        data = [list(state), action.squeeze(), list(next_state), reward.squeeze()]
        names = ['state', 'action', 'next_state', 'reward']
        replay_df = pd.DataFrame.from_dict(dict(zip(names, data)), dtype=int)
        #tqdm.write(f'replay df before duplicate removal length: {len(replay_df)}, sample: {replay_df.sample(3)}')
        replay_df.drop_duplicates(keep='first', inplace=True)
        #tqdm.write(f'replay df after duplicate removal length: {len(replay_df)}, sample: {replay_df.sample(3)}')
        ## normalize reward to make them non-negative
        min_reward = min(replay_df['reward'])
        if min_reward < 0:
            tqdm.write(f'replay df rewards are made non-negative by adding constant {-min_reward}')
            replay_df.loc[:, 'reward'] = replay_df['reward'].apply(lambda v: -min_reward + v)

        return replay_df
    
    def _get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def _get_state_rep(self, state):
        if self.q_model:
          with torch.no_grad():
            self.q_model(torch.FloatTensor(state).reshape((-1, self.state_dim)).to(self.device))
            return self.activation['q2'].squeeze()
        else:
          return np.array(state)
                    
    ## Build ANN indexes for each action in buffer
    def _build_ann_indexes(self):
        assert self.replay_df is not None
        assert 'core_index' in self.replay_df
        
        ann_indexes = {}
        state_rep_dim = self.q_model.q2.out_features if self.q_model else self.state_dim ##Equal to the out dim of the penultimate layer of q_model #self.state_dim
        for a in range(self.num_actions):
            annoy_indexer = ANN(state_rep_dim, 'euclidean')
            ann_indexes[a] = annoy_indexer
            
        tqdm.write('Indexing core states ANN')
        for index, row in tqdm(self.replay_df.iterrows()):
            if row['core_index'] == -1: continue ## Indexing only the core states
            a = row['action']
            ann_indexes[a].index(index, self._get_state_rep(row['state']))
            
        for a in range(self.num_actions):
            ann_indexes[a].build() ## Building binary trees for NNs

        return ann_indexes
    
    ## find approximate diameter (= longest distance between any two points) of the core states
    def _get_core_diameter(self):
        tqdm.write('Finding diameter of the core states')
        #core = [self._get_state_rep(row['next_state']).cpu().detach().numpy() for index, row in tqdm(self.replay_df.iterrows())]

        core = []
        core_idx = []
        for index, row in tqdm(self.replay_df.iterrows()):
            if row['core_index'] != -1 and index not in core_idx:
                core_idx.append(index)
                #core.append(self._get_state_rep(row['state']).cpu().detach().numpy())
                core.append(self._get_state_rep(row['state']))
                
        # set of random points from within the core states used to find diameter
        test_points = random.sample(core, 3)
        diameter = np.amax(cdist(test_points, core, 'euclidean'))
        tqdm.write(f'The diameter = {diameter} from a core set of size: {len(core)}')
        return diameter
    
    def _add_core_state_column(self):
        assert self.core_states is not None
        assert self.replay_df is not None
        assert 'state' in self.replay_df
        
        core_index_col = []
        for index, row in tqdm(self.replay_df.iterrows()):
            state_id = state_to_id(row['state'])
            core_index_col.append(self.core_states.get(state_id, -1))
        
        #tqdm.write(f'Core index col: {core_index_col}')
        self.replay_df['core_index'] = core_index_col
        tqdm.write(f'df after core index: {self.replay_df.sample(3)}')
    
    ## from buffer D:[(si, ai, si', ri)], return [si']
    def _detect_core_states(self):
        assert self.replay_df is not None
        core_id_map = {}
        core_idx = 0
        core = [row['next_state'] for index, row in tqdm(self.replay_df.iterrows())]
        #print(f'Core states: {core}')
        for s in set(core):
            core_id_map[state_to_id(s)] = core_idx
            core_idx += 1
        
        tqdm.write(f'Detected core states: {len(core_id_map)}')
        return core_id_map
    
    def get_policies(self):        
        return self.solvers

if __name__ == "__main__":
    
    from discrete_BCQ import FC_Q

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #state = [[0., 0.], [1., 0.], [0., 0.], [2., 2.], [0., 0.], [2., 2.]]
    #action = [0, 0, 0, 0, 1, 1]
    #next_state = [[1., 0.], [2., 2.], [1., 0.], [2., 2.], [2., 2.], [2., 2.]]
    #reward = [0, 2, 0, 1, 2, 1]
    #done = [0, 1, 0, 1, 1, 1]

    ## Example from $3 in the paper
    state = [[6., 1.], [2., 3.], [1., 5.], [3., 3.], [0., 5.], [2., 3.]]
    action = [0, 1, 1, 0, 1, 0]
    next_state = [[2., 3.], [6., 1.], [3., 3.], [1., 5.], [2., 3.], [0., 5.]]
    reward = [4, 2, 2, 2, 2, 2]
    done = [0, 1, 0, 1, 0, 1]
    
    
    ## Example experience data from a two action cyclic policy for a fixed arrival rate
    state = [[3., 1.], [3., 2.], [6., 1.], [5., 2.], [8., 1.], [7., 2.], [10., 1.], [9., 2.], \
            [3., 1.], [6., 1.], [5., 2.], [8., 1.], [7., 2.], [10., 1.], [9., 2.], [12., 1.], \
            ]
    action = [0, 1, 0, 1, 0, 1, 0, 1, \
             1, 0, 1, 0, 1, 0, 1, 0, \
             ]
    next_state = [[3., 2.], [6., 1.], [5., 2.], [8., 1.], [7., 2.], [10., 1.], [9., 2.], [12., 1.], \
                 [6., 1.], [5., 2.], [8., 1.], [7., 2.], [10., 1.], [9., 2.], [12., 1.], [11., 2.], \
                 ]
    reward = [3, 2, 4, 2, 4, 2, 4, 2, \
             1, 4, 2, 4, 2, 4, 2, 4, \
             ]
    done = [0, 0, 0, 0, 0, 0, 0, 1, \
           0, 0, 0, 0, 0, 0, 0, 1, \
           ]
    
    
    ## State(s) to evaluate trained policy built on the experience data above,
    ## Use the following config: mode=number, k=2, diameter=7, alpha=0.8, state rep=counts
    eval_state = [[4., 4.]] ## sequence got: (3,1)->(6,1)->(5,2)->(4,3)->(7,1)->(6,2)->(5,3)->(4,4)->(7,1)

    from utils import ReplayBuffer
    buffer = ReplayBuffer(state_dim=2, is_atari=False, atari_preprocessing=None, batch_size=4, buffer_size=len(state), device=device)
    for (s, a, n, r, d) in zip(state, action, next_state, reward, done):
        buffer.add(s, a, n, r, d, 0, 0)
    
    q_model = FC_Q(2, 2).to(device)
    dac = dac_builder(2, 2, buffer, q_model, device, 1, 'number', 7, 0.99, 0.01)
    policies = dac.get_policies()
    for s in state:
        a = policies[0].select_action(s)
        print(f'Chose action {a} for state {s}')
    for s in eval_state:
        a = policies[0].select_action(s)
        print(f'Chose action {a} for state {s}')
    #policies[0].print_stat()
    
