from wind_farm_gym import WindFarmEnv
import numpy as np
# Initialize the environment with 3 turbines positioned 750 meters apart in a line
#env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]))
#env = WindFarmEnv(turbine_layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250], [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]))
amalia_easting = [584022,583071,583532,583994,584455,582103,582570,583037,583503,583970,584437,584904,585371,581585.1,582057,582529,583002,583474,583946,584418,584890,585362,581068.1,581545,582024,582500,582978,583455,583932,584410,584887,581041.1,581523,582005.1,582488.1,582970,583452,583935,584417,584900,580531.1,581019.1,581506.1,581993,582481,582968,583455,583942,584430,580527.1,581019.1,581511,582002.1,582494,582986,583478,580547.1,581043.1,581539.1,582035.1]
amalia_northing = [5829007,5829056,5828757,5828458,5828159,5829063,5828772,5828481,5828191,5827900,5827608,5827318,5827027,5828734,5828452,5828170,5827888,5827606,5827323,5827041,5826759,5826477,5828385,5828111,5827839,5827566,5827293,5827020,5826747,5826473,5826200,5827763,5827499,5827235,5826971,5826707,5826443,5826179,5825915,5825651,5827405,5827150,5826895,5826640,5826385,5826130,5825875,5825620,5825365,5826802,5826556,5826310,5826064,5825818,5825571,5825325,5826228,5825990,5825752,5825515]
#env = WindFarmEnv(turbine_layout=(amalia_easting, amalia_northing))
#env = WindFarmEnv(turbine_layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250], [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]))
#env = WindFarmEnv(turbine_layout=([0], [0]))
env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]))

obs = env.reset()
for x in range(1000):                # Repeat for 1000 steps
    a = env.action_space.sample()    # Choose an action randomly
    a = np.zeros(env.n_turbines)
    #a = [-1]
    if x % 100 == 0:
        new_obs_list, reward_list, done, _ = env.MAstep(a)  # Perform the action
    print(sum(reward_list)) # / (1.0e-6 * 1 / 3600))
    env.render()                     # Render the environment; remove this line to speed up the process
    #print(reward / env._reward_scaling_factor)
    #print(reward)
    

env.close()