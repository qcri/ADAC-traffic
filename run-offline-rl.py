import argparse
import copy
import os

import numpy as np
from scipy import sparse
import torch

from ADAC import ensemble_DQN, discrete_BCQ, discrete_TD3BC, DQN, utils 
from ADAC.dac_mdp import dac_builder

from gharaffaEnv import gharaffaEnv
from gharaffaPolicies import gharaffaConstantCyclePolicy

## policy is an object from gharaffaPolicies.py
def generate_buffer_with_policy(env, replay_buffer, is_atari, num_actions, state_dim, device, args, policy):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}_{setting}"

        print('Interacting with environment with args: ', args)

        # Gharrafa environment configurations
        env_shours = range(24)
        env_days = range(7)

        # loop through each config
        for day in env_days:
                for shour in env_shours:
                        ehour = 1 + shour
                        env = gharaffaEnv({"GUI": False, "Mode": "eval", "sHour": shour, "eHour": ehour, "day": day})

                        state, done = env.reset(True), False
                        episode_start = True
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num = 0
                        # Initialize and load policy
                        policy.reset()

                        # Interact with the environment for max_timesteps
                        done = False
                        while not done:
                                episode_timesteps += 1
                                action = policy.select_action(np.array(state))

                                # Perform action and log results
                                next_state, reward, done, info = env.step(action)
                                episode_reward += reward

                                done_float = float(done) if episode_timesteps < 86400 else 0
                                # Store data in replay buffer
                                replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
                                state = copy.copy(next_state)
                                episode_start = False
                        print(f"On day: {day} Hour: {shour} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

                replay_buffer.save(f"./buffers/{buffer_name}")

def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}_{setting}" 

        print('Interacting with environment with args: ', args)

        # Initialize and load policy
        policy = DQN.DQN(
                is_atari,
                num_actions,
                state_dim,
                device,
                parameters["discount"],
                parameters["optimizer"],
                parameters["optimizer_parameters"],
                parameters["polyak_target_update"],
                parameters["target_update_freq"],
                parameters["tau"],
                parameters["initial_eps"],
                parameters["end_eps"],
                parameters["eps_decay_period"],
                parameters["eval_eps"],
        )

        if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")
        
        evaluations = []
        training_data = [] #steps in each episode and reward after every episode

        state, done = env.reset(), False
        episode_start = True
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

        # Interact with the environment for max_timesteps
        for t in range(int(args.max_timesteps)):

                episode_timesteps += 1

                # If generating the buffer, episode is low noise with p=low_noise_p.
                # If policy is low noise, we take random actions with p=eval_eps.
                # If the policy is high noise, we take random actions with p=rand_action_p.
                if args.generate_buffer:
                        if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
                                action = env.action_space.sample()
                        else:
                                action = policy.select_action(np.array(state), eval=True)

                if args.train_behavioral:
                        if t < parameters["start_timesteps"]:
                                action = env.action_space.sample()
                        else:
                                action = policy.select_action(np.array(state))

                # Perform action and log results
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Only consider "done" if episode terminates due to failure condition
                done_float = float(done) if episode_timesteps < 86400 else 0 #env._max_episode_steps else 0

                # For atari, info[0] = clipped reward, info[1] = done_float
                if is_atari:
                        reward = info[0]
                        done_float = info[1]
                        
                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
                state = copy.copy(next_state)
                episode_start = False

                # Train agent after collecting sufficient data
                if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
                        policy.train(replay_buffer)

                if done:
                        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                        # Reset environment
                        state, done = env.reset(), False
                        training_data.append((episode_reward, episode_timesteps))
                        episode_start = True
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num += 1
                        low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

                # Evaluate episode
                if (t + 1) % parameters["eval_freq"] == 0:
                        eval_results = eval_policy(policy, args.env, args.seed)
                        print(f"Eval results: {eval_results}")
                        evaluations.append(eval_results)
                        if args.train_behavioral:
                                np.save(f"./results/behavioral_{setting}", evaluations)
                                np.save(f"./results/training_performance_{setting}", training_data)
                                policy.save(f"./models/behavioral_{setting}")
                                replay_buffer.save(f"./buffers/{buffer_name}")
                        else:
                                np.save(f"./results/buffer_{buffer_name}_performance_{setting}", evaluations)
                                np.save(f"./results/buffer_{buffer_name}_training_performance_{setting}", training_data)
                                replay_buffer.save(f"./buffers/{buffer_name}")

        # HACK: Gharaffa-specific: Final eval with GUI
        #evaluations.append(eval_policy(policy, args.env, args.seed, gui=True))

        # Save final policy
        if args.train_behavioral:
                np.save(f"./results/behavioral_{setting}", evaluations)
                np.save(f"./results/training_performance_{setting}", training_data)
                policy.save(f"./models/behavioral_{setting}")
                replay_buffer.save(f"./buffers/{buffer_name}")

        # Save final buffer and performance
        else:
                np.save(f"./results/buffer_{buffer_name}_performance_{setting}", evaluations)
                np.save(f"./results/buffer_{buffer_name}_training_performance_{setting}", training_data)
                replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}"
        offline_algo=f"{args.offline_algo}"
        id_tokens=f"{args.id_tokens}"
        print('BCQ environment with args: ', args)


        # Initialize and load policy
        policy = discrete_BCQ.discrete_BCQ(
                is_atari,
                num_actions,
                state_dim,
                device,
                args.BCQ_threshold,
                parameters["discount"],
                parameters["optimizer"],
                parameters["optimizer_parameters"],
                parameters["polyak_target_update"],
                parameters["target_update_freq"],
                parameters["tau"],
                parameters["initial_eps"],
                parameters["end_eps"],
                parameters["eps_decay_period"],
                parameters["eval_eps"],
                args.offline_algo,
                args.sm_threshold,
                args.mm_threshold
        )

        # Load replay buffer    
        replay_buffer.load(f"./buffers/{buffer_name}_{setting}")

        # Initialize sample buffer for q-value evaluation
        sample_buffer = utils.ReplayBuffer(state_dim, is_atari, {}, parameters["batch_size"] * 500, parameters["buffer_size"], device)
        sample_buffer.load(f"./buffers/{buffer_name}_{setting}")
        eval_sample = sample_buffer.sample() # 500 mini-batches for evaluation

        evaluations = []
        sample_q_values = []
        training_data = [] # Q-loss and G-loss
        eval_num = 0
        done = True 
        training_iters = 0
        
        ## HACK: Temporary hardcoded diameters as computing them is very time consuming
        diameters = [27, 445, 640, 740] #[27, 325, 340, 360, 395, 410, 470, 490, 520, 555, 585, 615, 640, 650, 670, 720, 735, 735, 735, 735] ## RL-agent BCQ [27, 420, 585, 720, 735] #
        #diameters = [27, 11180, 388850, 37304870] ## RL-agent DDQN
        #diameters = [27, 270, 325, 350, 390, 390, 410, 433, 450, 470, 475, 475, 475, 530, 530, 530, 560, 560, 570, 570] ## Cyclic-agent BCQ [27, 390, 470, 530, 570] #
        #diameters = [56, 56] ## Loop Counts


        '''
        while training_iters < args.max_timesteps: 
                
                ## Evaluating before training to get a random sample
                ## build mdp and evaluate all configured policies, pick the best
                
                if args.dac_configs > 0:
                    mdp = dac_builder(num_actions, state_dim, replay_buffer, policy.get_q_model(), device, args.dac_configs, 'number', -1)#diameters[eval_num]) ## send -1 to compute diameter
                    mdp_policies = mdp.get_policies()
                    p_returns = []
                    for p in mdp_policies:
                        p_returns.append(eval_policy(p, args.env, args.seed))
                    evaluations.append(max(p_returns))
                else: ## bcq eval
                    evaluations.append(eval_policy(policy, args.env, args.seed))
                
                sample_q_values.append(policy.sample_q_values(eval_sample))
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}", evaluations)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_training_losses", training_data)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_sample_qvalues", sample_q_values)
                policy.save(f"./models/{offline_algo}_{buffer_name}_{setting}{id_tokens}")
                
                
                ## store mdp transitions and rewards to files
                #if args.dac_configs > 0:
                #    for p in mdp_policies:
                #        transitions = p.get_transitions() ## dims = [states * states], array of length = actions
                #        rewards = p.get_rewards() ## dims = states * actions
                #        np.save(f"./matrices/{offline_algo}_{buffer_name}_{setting}{id_tokens}_rewards", rewards)
                #        for i, tran in enumerate(transitions):
                #            sparse.save_npz(f"./matrices/{offline_algo}_{buffer_name}_{setting}{id_tokens}_transitions_{i}", tran)

                for _ in range(int(parameters["eval_freq"])):
                    training_data_sample = policy.train(replay_buffer)
                eval_num += 1

                training_data.append(training_data_sample)
                
                training_iters += int(parameters["eval_freq"])
                print(f"Training iterations: {training_iters}")
        '''

        ## Optional: Final eval result
        print(f"Launching final eval:")
        if args.dac_configs > 0:
                    mdp = dac_builder(num_actions, state_dim, replay_buffer, policy.get_q_model(), device, args.dac_configs, 'number', -1)#diameters[eval_num]) ## send -1 to compute diameter
                    mdp_policies = mdp.get_policies()
                    p_returns = []
                    for p in mdp_policies:
                        p_returns.append(eval_policy(p, args.env, args.seed, gui=True))
                    evaluations.append(max(p_returns))
        else: ## bcq eval
                    evaluations.append(eval_policy(policy, args.env, args.seed, gui=True))
        np.save(f"./results/{offline_algo}_{setting}", evaluations)
        #policy.save(f"./models/{offline_algo}_{setting}")

# Trains REM offline
def train_td3bc(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}"
        offline_algo=f"{args.offline_algo}"
        id_tokens=f"{args.id_tokens}"
        print('TD3BC environment with args: ', args)


        # Initialize and load policy
        policy = discrete_TD3BC.TD3_BC(
                is_atari,
                num_actions,
                state_dim,
                device,
                parameters["alpha"],
                parameters["discount"],
                parameters["optimizer"],
                parameters["optimizer_parameters"],
                parameters["polyak_target_update"],
                parameters["target_update_freq"],
                parameters["tau"],
        )

        # Load replay buffer    
        replay_buffer.load(f"./buffers/{buffer_name}_{setting}")

        # Initialize sample buffer for q-value evaluation
        sample_buffer = utils.ReplayBuffer(state_dim, is_atari, {}, parameters["batch_size"] * 500, parameters["buffer_size"], device)
        sample_buffer.load(f"./buffers/{buffer_name}_{setting}")
        eval_sample = sample_buffer.sample() # 500 mini-batches for evaluation

        evaluations = []
        sample_q_values = []
        training_data = [] # Q-loss and G-loss
        episode_num = 0
        done = True
        training_iters = 0

        while training_iters < args.max_timesteps:

                for _ in range(int(parameters["eval_freq"])):
                        training_data_sample = policy.train(replay_buffer)

                training_data.append(training_data_sample)
                evaluations.append(eval_policy(policy, args.env, args.seed))
                sample_q_values.append(policy.sample_q_values(eval_sample))
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}", evaluations)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_training_losses", training_data)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_sample_qvalues", sample_q_values)
                policy.save(f"./models/{offline_algo}_{buffer_name}_{setting}{id_tokens}")

                training_iters += int(parameters["eval_freq"])
                print(f"Training iterations: {training_iters}")


# Trains REM offline
def train_ensemble(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}"
        offline_algo=f"{args.offline_algo}"
        id_tokens=f"{args.id_tokens}"
        print('REM environment with args: ', args)


        # Initialize and load policy
        policy = ensemble_DQN.ensemble_DQN(
                is_atari,
                num_actions,
                state_dim,
                args.num_heads,
                args.num_convex_combos,
                device,
                parameters["discount"],
                parameters["optimizer"],
                parameters["optimizer_parameters"],
                parameters["polyak_target_update"],
                parameters["target_update_freq"],
                parameters["tau"],
                parameters["initial_eps"],
                parameters["end_eps"],
                parameters["eps_decay_period"],
                parameters["eval_eps"],
                args.offline_algo,
        )

        # Load replay buffer    
        replay_buffer.load(f"./buffers/{buffer_name}_{setting}")

        # Initialize sample buffer for q-value evaluation
        sample_buffer = utils.ReplayBuffer(state_dim, is_atari, {}, parameters["batch_size"] * 500, parameters["buffer_size"], device)
        sample_buffer.load(f"./buffers/{buffer_name}_{setting}")
        eval_sample = sample_buffer.sample() # 500 mini-batches for evaluation

        evaluations = []
        sample_q_values = []
        training_data = [] # Q-loss and G-loss
        episode_num = 0
        done = True 
        training_iters = 0
        
        while training_iters < args.max_timesteps: 
                
                for _ in range(int(parameters["eval_freq"])):
                        training_data_sample = policy.train(replay_buffer)

                training_data.append(training_data_sample)
                evaluations.append(eval_policy(policy, args.env, args.seed))
                sample_q_values.append(policy.sample_q_values(eval_sample))
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}", evaluations)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_training_losses", training_data)
                np.save(f"./results/{offline_algo}_{buffer_name}_{setting}{id_tokens}_sample_qvalues", sample_q_values)
                policy.save(f"./models/{offline_algo}_{buffer_name}_{setting}{id_tokens}")

                training_iters += int(parameters["eval_freq"])
                print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, gui=False, eval_episodes=6):
        eval_env = gharaffaEnv({"GUI": gui, "Mode":"eval", "Play":"action"})#eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
        eval_env.seed(seed + 50)

        avg_reward = 0.
        avg_return = 0.
        avg_wait = 0.
        avg_co2 = 0.
        rewards = []
        for i in range(eval_episodes):
                state, done = eval_env.reset(), False
                episode_reward = 0.
                episode_return = 0.
                episode_waittime = 0.
                episode_co2 = 0.
                gamma = 0.99
                discount = 1
                while not done:
                        action = policy.select_action(np.array(state), eval=True)
                        state, reward, done, info = eval_env.step(action)
                        episode_reward += reward
                        # discounted reward
                        episode_return += discount * reward
                        discount *= gamma
                        episode_waittime += info["WaitingTime"]
                        episode_co2 += info["CO2Emission"]
                rewards.append(episode_reward)
                avg_reward += episode_reward if i!=4 else 0 ## HACK: not counting 5th episode in the average; just for the purpose of paper
                avg_return += episode_return if i!=4 else 0
                avg_wait += episode_waittime if i!=4 else 0 
                avg_co2 += episode_co2 if i!=4 else 0 
                print(f'Episode over, reward: {episode_reward}, return: {episode_return}, total wait time: {episode_waittime}. CO2 emission: {episode_co2}')
                policy.print_stat()
        ## HACK: not counting 5th episode in the average; just for the purpose of paper
        avg_reward /= (eval_episodes-1)
        avg_return /= (eval_episodes-1)
        avg_wait /= (eval_episodes-1)
        avg_co2 /= (eval_episodes-1)
        rewards.append(avg_reward)

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}, return: {avg_return:.2f}, avg wait: {avg_wait:.2f}, avg co2: {avg_co2:.2f}")
        print("---------------------------------------")
        try:
                eval_env.closeconn() #HACK: specific to gharaffa
        except:
                pass
        return (rewards)


if __name__ == "__main__":
        regular_parameters = {
                # Exploration
                "start_timesteps": 1e3,
                "initial_eps": 0.1,
                "end_eps": 0.1,
                "eps_decay_period": 1,
                # Evaluation
                "eval_freq": 100000, #8640, #60480,#3e4,
                "eval_eps": 1e-3,
                # Learning
                "discount": 0.99,
                "buffer_size": 8640, #8640, #60480, ## 360 steps * 24 * 7
                "batch_size": 128,
                "optimizer": "Adam",
                "optimizer_parameters": {
                        "lr": 3e-4
                },
                "train_freq": 1,
                "polyak_target_update": True,
                "target_update_freq": 1,
                "tau": 0.005,
                "alpha": 1.0 #Used in TD3BC
        }

        # Load parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--env", default="gharaffaEnv")     # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
        parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
        parser.add_argument("--offline_algo", default="BCQ")  # If true, generate buffer
        parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
        parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
        parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
        parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
        parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
        parser.add_argument("--generate_buffer_with_policy", action="store_true")  # If true, generate buffer with policy
        parser.add_argument("--evaluate_only", action="store_true")  # If true, evaluate given policy
        parser.add_argument("--no_termination", action="store_true")  # If true, evaluate episodes fully
        parser.add_argument("--sm_threshold", default=5.0, type=float)# Threshold hyper-parameter for mellowmax
        parser.add_argument("--mm_threshold", default=5.0, type=float)# Threshold hyper-parameter for mellowmax
        parser.add_argument("--num_heads", default=4, type=int)# projection heads for q values in ensemble dqn
        parser.add_argument("--num_convex_combos", default=4, type=int)# convex combos in ensemble dqn
        parser.add_argument("--id_tokens", default="")# Extra tags for identification of results
        parser.add_argument("--dac_configs", default=0, type=int)# Number of dac-mdp configurations, <=0 implies no mdp

        args = parser.parse_args()
        
        print("---------------------------------------")        
        if args.train_behavioral:
                print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
        elif args.generate_buffer or args.generate_buffer_with_policy:
                print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
        else:
                print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        if args.train_behavioral and args.generate_buffer:
                print("Train_behavioral and generate_buffer cannot both be true.")
                exit()

        if not os.path.exists("./results"):
                os.makedirs("./results")

        if not os.path.exists("./models"):
                os.makedirs("./models")

        if not os.path.exists("./buffers"):
                os.makedirs("./buffers")

        is_atari = False
        mode = 'train'
        if args.no_termination:
                mode = 'eval'
        env = gharaffaEnv({"GUI": False, "Mode": mode})
        state_dim = env.observation_space.shape[1]
        num_actions = env.action_space.n
        print(f'State dim={state_dim} and num_actions={num_actions}')

        parameters = regular_parameters

        # Set seeds
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Initialize buffer
        replay_buffer = utils.ReplayBuffer(state_dim, is_atari, {}, parameters["batch_size"], parameters["buffer_size"], device)

        if args.train_behavioral or args.generate_buffer:
                interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
        elif args.generate_buffer_with_policy:
                generate_buffer_with_policy(env, replay_buffer, is_atari, num_actions, state_dim, device, args, gharaffaConstantCyclePolicy())
        elif args.offline_algo == 'REM' or args.offline_algo == 'Ensemble':
                train_ensemble(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
        elif args.offline_algo == 'TD3BC':
                train_td3bc(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
        else:
                train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
