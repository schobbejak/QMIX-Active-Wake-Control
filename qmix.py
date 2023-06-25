import argparse
import time
import datetime
import numpy as np 
import torch
import os
import itertools
import csv
from copy import deepcopy
from agent.QMIX import ReplayMemory
from agent.QMIX import AgentsTrainer
from wind_farm_gym import WindFarmEnv
from agent import NaiveAgent, FlorisAgent, SACAgent, TD3Agent
import matplotlib.pyplot as plt
#import multiagent.scenarios as scenarios

parser = argparse.ArgumentParser(description='PyTorch QMIX Args')
parser.add_argument('--algorithm', type=str, default='qmix', help='algorithm to run')
parser.add_argument('--environment', type=str, default='3', help='environment')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=25, help='maximum episode length')
parser.add_argument('--policy_lr', type=float, default=0.00005, help='learning rate for policies')
parser.add_argument('--critic_lr', type=float, default=0.0005, help='learning rate for critics')
parser.add_argument('--alpha', type=float, default=0.01, help='policy entropy term coefficient')
parser.add_argument('--tau', type=float, default=0.05, help='target network smoothing coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 16)') # episodes
parser.add_argument('--hidden_dim', type=int, default=128, help='network hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=1000, help='steps before training begins')
parser.add_argument('--target_update_interval', type=int, default=5, help='target network update interval')
parser.add_argument('--updates_per_step', type=int, default=5, help='network update frequency')
parser.add_argument('--replay_size', type=int, default=500000, help='maximum number of episodes of replay buffer')
parser.add_argument('--cuda', action='store_false', help='run on GPU (default: False)')
parser.add_argument('--render', action='store_true', help='render or not')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load environment
#scenario = scenarios.load(args.scenario + '.py').Scenario()
#world = scenario.make_world()
#env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete_action_space=False)
amalia_easting = [584022,583071,583532,583994,584455,582103,582570,583037,583503,583970,584437,584904,585371,581585.1,582057,582529,583002,583474,583946,584418,584890,585362,581068.1,581545,582024,582500,582978,583455,583932,584410,584887,581041.1,581523,582005.1,582488.1,582970,583452,583935,584417,584900,580531.1,581019.1,581506.1,581993,582481,582968,583455,583942,584430,580527.1,581019.1,581511,582002.1,582494,582986,583478,580547.1,581043.1,581539.1,582035.1]
amalia_northing = [5829007,5829056,5828757,5828458,5828159,5829063,5828772,5828481,5828191,5827900,5827608,5827318,5827027,5828734,5828452,5828170,5827888,5827606,5827323,5827041,5826759,5826477,5828385,5828111,5827839,5827566,5827293,5827020,5826747,5826473,5826200,5827763,5827499,5827235,5826971,5826707,5826443,5826179,5825915,5825651,5827405,5827150,5826895,5826640,5826385,5826130,5825875,5825620,5825365,5826802,5826556,5826310,5826064,5825818,5825571,5825325,5826228,5825990,5825752,5825515]
#env = WindFarmEnv(turbine_layout=(amalia_easting, amalia_northing))
#env = WindFarmEnv(turbine_layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250], [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]))
env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]))
#env = WindFarmEnv(turbine_layout=([0, 750], [0, 0]))

if args.environment == "3":
    env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]), max_angular_velocity=2.0, desired_yaw_boundaries=(-30.0, 30.0))
elif args.environment == "16":
    env = WindFarmEnv(turbine_layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250], [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]), max_angular_velocity=2.0, desired_yaw_boundaries=(-30.0, 30.0))
elif args.environment == "amalia":
    env = WindFarmEnv(turbine_layout=(amalia_easting, amalia_northing), max_angular_velocity=2.0, desired_yaw_boundaries=(-30.0, 30.0))

# Shapes
obs_shape = 3
action_shape = 1

def calcProcessTime(starttime, cur_iter, max_iter):
    curr_time = time.time()
    cur_iter = cur_iter - (curr_batchsize * 2)
    if cur_iter <= 0:
        cur_iter = 1
    max_iter = max_iter - (curr_batchsize * 2)
    telapsed = curr_time - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time
    currtime = datetime.datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime, currtime)

memory = ReplayMemory(args.replay_size, args.max_episode_len, env.n_turbines, obs_shape, action_shape)
trainer = AgentsTrainer(env.n_turbines, obs_shape, action_shape, args)
total_numsteps = 0
updates = 0
t_start = time.time()
reward_bias = 0.
train_policy = False

agentTD3 = TD3Agent("td3", env)

# Retrieve baseline power
_, baselinePower, _, _ = env.step(np.zeros(env.n_turbines))
baselinePower = baselinePower
baseLine = [baselinePower for x in range(args.num_episodes + 1)]
resultsPlot = [baselinePower]

# Logging data
header = ['episode', 'avg_power']
resultsPath = "windfarm/results/" + "algorithm_" + args.algorithm + "_n_turbines_" + str(env.n_turbines) + "_num_episodes_" + str(args.num_episodes) + "_max_episode_len_" + str(args.max_episode_len) + "_policy_lr_" + str(args.policy_lr) + "_critic_lr_" + str(args.critic_lr) + "_alpha_" + str(args.alpha) + "_tau_" + str(args.tau) + "_gamma_" + str(args.gamma) + "_seed_" + str(args.seed) + "_batch_size_" + str(args.batch_size) + "_hidden_dim_" + str(args.hidden_dim) + "_target_update_interval_" + str(args.target_update_interval) + "_updates_per_step_" + str(args.updates_per_step) + ".csv"
resultsFile = open(resultsPath, 'w')
resultsWriter = csv.writer(resultsFile)
resultsWriter.writerow(header)

for i_episode in itertools.count(1):
    
    episode_reward = 0.0 # sum of all agents
    episode_reward_per_agent = [0.0 for _ in range(env.n_turbines)] # reward list
    step_within_episode = 0.0
    obs_list = env.resetMA()
    obs = env.reset()

    if args.algorithm == "qmix":
        
        done = False
        while not done:
            obs_array = np.asarray(obs_list)
            action_list = trainer.act(obs_array)
            action_list = list(action_list)
            action_list = [x[0] for x in action_list]

            # interact with the environment
            new_obs_list, reward_list, done, _ = env.MAstep(action_list)
            #if args.render:
            #env.render()
            total_numsteps += 1
            step_within_episode += 1
            all_done = done #all(done_list)
            terminated = (step_within_episode >= args.max_episode_len)
            done = all_done or terminated
            #print(sum(reward_list) / env._reward_scaling_factor)
            # replay memory filling
            memory.push(np.asarray(obs_list), [[x] for x in action_list], sum(reward_list) + reward_bias, np.asarray(new_obs_list),
                        1. if (all_done and not terminated) else 0.)
            # memory.push(np.asarray(obs_list), np.asarray(action_list), reward_list[0], np.asarray(new_obs_list),
            #              1. if done else 0.)
            obs_list = new_obs_list

            episode_reward += sum(reward_list)
            for i in range(len(episode_reward_per_agent)):
                episode_reward_per_agent[i] += reward_list[i]

        memory.end_episode()
        trainer.reset()

        curr_batchsize = args.batch_size
        if i_episode > 10:
            train_policy = True
        if i_episode == curr_batchsize * 2:
            t_start = time.time()
        if len(memory) > curr_batchsize * 2:
            for _ in range(args.updates_per_step):
                obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, done_batch = memory.sample(curr_batchsize)
                sample_batch = (obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, done_batch)
                trainer.update_parameters(sample_batch, curr_batchsize, updates, train_policy)
                updates += 1
    elif args.algorithm == "td3":
        done = False
        while not done:
            action = agentTD3.find_action(obs)
            old_obs = obs
            obs, reward, _, _ = env.step(action)
            agentTD3.learn(old_obs, action, reward, obs, i_episode)
            step_within_episode += 1
            done = step_within_episode >= args.max_episode_len
            episode_reward += reward


    avg_power = round(episode_reward / args.max_episode_len, 10)
    #print("Episode: {}, total steps: {}, total episodes: {}, avg power: {}".format(i_episode, total_numsteps,step_within_episode, avg_power))
    resultsPlot.append(avg_power)
    if i_episode % 100 == 0:
        #print(i_episode / args.num_episodes)
        print("Epoch: " + str(i_episode) + ", Average power: " + str(avg_power) + ", Batch size: " + str(args.batch_size))
        #print("Actions summary:" + str(actions_summary))
        print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s, current time: %s"%calcProcessTime(t_start,i_episode ,args.num_episodes))
        plt.plot(resultsPlot)
        plt.plot(baseLine)
        plt.xlabel("Episode")
        plt.ylabel("Average power output per second (MWh)")
        plt.title("Power output vs episode for " + str(env.n_turbines) + " turbine windfarm")
        strFile = "windfarm/figs/" + "algorithm_" + args.algorithm + "_n_turbines_" + str(env.n_turbines) + "_num_episodes_" + str(args.num_episodes) + "_max_episode_len_" + str(args.max_episode_len) + "_policy_lr_" + str(args.policy_lr) + "_critic_lr_" + str(args.critic_lr) + "_alpha_" + str(args.alpha) + "_tau_" + str(args.tau) + "_gamma_" + str(args.gamma) + "_seed_" + str(args.seed) + "_batch_size_" + str(args.batch_size) + "_hidden_dim_" + str(args.hidden_dim) + "_target_update_interval_" + str(args.target_update_interval) + "_updates_per_step_" + str(args.updates_per_step) + ".png"
        plt.savefig(strFile)
        plt.close()
    
    if i_episode >= args.num_episodes:
        break

    # Write to csv file
    resultsWriter.writerow([i_episode, avg_power])

# Close items
env.close()
resultsFile.close()


# Create plot
axisX = [x for x in range(args.num_episodes + 1)]
#fig, (ax1, ax2) = plt.subplots(1, 2)
#fig.suptitle('Plots')
plt.plot(axisX, resultsPlot)
plt.plot(axisX, baseLine)
plt.xlabel("Episode")
plt.ylabel("Average power output per second (MWh)")
plt.title("Power output vs episode for " + str(env.n_turbines) + " turbine windfarm")
#ax2.plot(criticLoss)
#ax2.plot(policyLoss)
strFile = "windfarm/figs/" + "algorithm_" + args.algorithm + "_n_turbines_" + str(env.n_turbines) + "_num_episodes_" + str(args.num_episodes) + "_max_episode_len_" + str(args.max_episode_len) + "_policy_lr_" + str(args.policy_lr) + "_critic_lr_" + str(args.critic_lr) + "_alpha_" + str(args.alpha) + "_tau_" + str(args.tau) + "_gamma_" + str(args.gamma) + "_seed_" + str(args.seed) + "_batch_size_" + str(args.batch_size) + "_hidden_dim_" + str(args.hidden_dim) + "_target_update_interval_" + str(args.target_update_interval) + "_updates_per_step_" + str(args.updates_per_step) + ".png"
plt.savefig(strFile)
plt.show()
