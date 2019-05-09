import gym
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env = env.unwrapped


print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# adjust hyper-parameters here
RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.02,output_graph=False)

total_steps = 0
steps_list=list()
record=0
for i_episode in range(50):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        # this is another reward
        #if done:
        #    reward = 100     # r in [0, 1]
        #else:
        #    reward = -0.5
        #reward=abs(position-(-0.5))
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            steps_list.append(total_steps-record)
            record=total_steps
            break

        observation = observation_
        total_steps += 1
        

env.close()
#RL.plot_cost()
#plt.plot(np.arange(200),steps_list)
#plt.xlabel('Episode No.')
#plt.ylabel('Step of episode')
