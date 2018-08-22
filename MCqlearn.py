import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
#from MountainCar import MountainCar
import gym 

class ExperienceReplay(object):
    def __init__(self, max_memory=100, max_good_memory = 100, discount=.9):
        self.max_memory = max_memory
        self.max_good_memory = max_good_memory
        self.memory_good = list()
        self.memory_rand = list()
        self.memory = list
        self.discount = discount

    def remember(self, states, game_over,percentage):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory_rand.append([states, game_over])
        if len(self.memory_rand) > self.max_memory:
            del self.memory_rand[0]
            
        if game_over:
            self.memory_good += self.memory_rand[-self.max_good_memory:]
            
            if len(self.memory_good) > self.max_memory:
                self.memory_good = random.sample(self.memory_good,self.max_memory)
                                
        if len(self.memory_good) == 0:
            self.memory_good = self.memory_rand
        
        if len(self.memory_good) == self.max_memory:
             self.memory_rand = self.memory_good
            
        sample_good = random.choices(self.memory_good, k = int(self.max_memory*percentage))
        sample_rand = random.choices(self.memory_rand, k = int(self.max_memory*(1-percentage)))
        
        self.memory = sample_good+sample_rand

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        # env_dim = self.memory[0][0][0].shape[1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

if __name__ == "__main__":
    # parameters
    epsilon_abs = 9
    #epsilon = .1
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 30
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    input_size = 2

    Xrange = [-1.5, 0.55]
    Vrange = [-0.7, 0.7]
    start = [-0.5, 0.0]
    goal = [0.45]

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(2, ), activation='relu'))
    # model.add(Dense(hidden_size, input_shape=(2,1)))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line below
    # model.load_weights("model.h5")

    # Define environment/game
    env = gym.make('MountainCar-v0')

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        percentage = 0.5 + e/(2*epoch)
        epsilon = max(.1,epsilon_abs-e/(2*epsilon_abs+1))  # exploration
        loss = 0.
        done = False
        # get initial input
        input_t = env.reset()
        input_t = np.array([input_t[0],input_t[1]]).reshape((1,-1))
        step = 0
        while (not done):
            env.render()
            input_tm1 = input_t
            step += 1
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)[0]
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, done, info = env.step(action)
            
#            if input_t[0] >=-0.25:
#                reward += 1
#            
#            if input_t[0] >=0.:
#                reward += 2

            if input_t[0] >=0.5:
                win_cnt += 1
            else:
                done = False

            input_t = input_t.reshape((1,-1))
            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], done, percentage)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

        print("Step {} Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(step, e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)