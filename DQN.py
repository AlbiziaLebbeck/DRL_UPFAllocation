import numpy as np
from multiprocessing import Process, Queue, Barrier, Lock
from collections import deque
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, RMSprop

from NetworkEnv import NetworkEnv
from ModifiedTensorBoard import ModifiedTensorBoard
import time
import random


class DQNAgent:
    def __init__(self, env, num_upf):
        self.state_size = env.observation_spec().shape
        self.action_size = env.action_spec().shape[0]
        self.num_upf = num_upf

        # Hyperparameters
        self.gamma = 0.99            # Discount rate
        self.lr =1e-3
        self.epsilon = 1
        self.epsilon_decay = 0.9
        self.min_epsilon = 0
        self.num_learn = 0
        self.update_rate = 1

        # Construct Actor and Critic models
        self.Qs = self._build_model()
        self.targetQs = self._build_model()
        for i in range(self.num_upf):
            self.targetQs[i].set_weights(self.Qs[i].get_weights())

    # Constructs NN
    def _build_model(self):
        input_state = Input(self.state_size)
        d1 = Dense(1024, activation='relu')(input_state)
        d2 = Dense(1024, activation='relu')(d1)
        q = [Dense(self.action_size, activation='linear')(d2) for _ in range(self.num_upf)]

        Qs = []
        for i in range(self.num_upf):
            Q = Model(inputs=input_state, outputs=q[i])
            Q.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
            Qs.append(Q)

        return Qs

    def action(self, state):
        actions = []

        for i in range(self.num_upf):
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                action = np.argmax(self.Qs[i](np.array([state])))
            actions.append(action)
        
        return actions

    def learn(self, states, actions, next_states, rewards):

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        
        num_data = len(states)

        loss = 0
        for i in range(self.num_upf):
            targets = np.array(self.Qs[i](states))
            next_Qvalues = self.targetQs[i](next_states)
            targets[range(num_data),actions[:,i]] = rewards + self.gamma * np.amax(next_Qvalues, axis=1)
        
            hist = self.Qs[i].fit(states, targets, epochs=1, verbose=0)
            loss += hist.history['loss'][0]

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon

        self.num_learn += 1
        if self.num_learn % self.update_rate == 0:
            for i in range(self.num_upf):
                self.targetQs[i].set_weights(self.Qs[i].get_weights())

        return loss

    def load(self, name):
        for i in range(self.num_upf):
            self.Qs[i] = load_model('models/' + name + '_Q'+str(i)+'.h5', compile=False)

    def save(self, name):
        for i in range(self.num_upf):
            self.Qs[i].save('models/' + name + '_Q'+str(i)+'.h5')


if __name__ == '__main__':
    for speeds in [1,2,3]:
        ue_file = "koln.tr"#args.ueFile
        bs_file = "koln_bs-deployment-D1.log"#args.bsFile
        episodes = 50
        batch_size = 100
        evaluate = False

        # global_env = NetworkEnv(bs_file, ue_file)
        # global_agent = ACAgent(global_env)

        best_score = -float('inf')

        env = NetworkEnv(bs_file, ue_file, speeds)
        
        a1 = 0
        a2 = 1
        a3 = 0

        # name = "new_a2c5nodes-{}".format(int(time.time()))
        name = "dqn78_{:d}spd_{:03d}_{:03d}_1".format(speeds, int(a2*100), int(a3*100))
        
        agent = DQNAgent(env, 1)
        if evaluate:
            agent.load(name)
            # agent.summary()
            agent.epsilon = 0
            agent.tensorboard = ModifiedTensorBoard("ac5nodes", log_dir="logs_{}nodes/{}".format(env.N, name + '_eval'))
        else:
            agent.tensorboard = ModifiedTensorBoard("ac5nodes", log_dir="logs_{}nodes/{}".format(env.N, name))

        # if e % 10 == 0:
        #     all_states = []
        #     all_actions = []
        #     all_returns = []
        
        # states = deque(maxlen=1000)
        # actions = deque(maxlen=1000)
        # next_states = deque(maxlen=1000)
        # rewards = deque(maxlen=1000)
        memory = deque(maxlen=150)

        num_step = 0
        
        state = env.reset()
        
        for iteration in range(16560): #8640

            save_medel = True

            # Transition Dynamics
            action = agent.action(state)
            upf_placement = [0] * agent.action_size
            for a in action:
                upf_placement[a] = 1
            
            start_time = time.process_time()
            next_state, reward, done, timestamp = env.step(upf_placement)
            end_time = time.process_time()

            if done:
                break 
            
            agent.tensorboard.step = iteration + 1
            
            penalty_num_upf = reward[0] 
            penalty_latency = reward[1]
            penalty_migration = reward[2]
            # total_reward = - a1 * penalty_num_upf/env.N - a2*penalty_latency/env.max_num_hops - a3*penalty_migration / (len(env.prev_UEs) * env.max_num_hops)
            total_reward = -penalty_latency
            
            # states.append(state)
            # actions.append(action)
            # next_states.append(next_state)
            # rewards.append(total_reward)
            memory.append([state,action,next_state,total_reward])

            for spd in range(speeds):
                num_ue_spd = sum(state[spd*env.N:(spd+1)*env.N])
                agent.tensorboard.update_stats(**{"numUEs_s"+str(spd):num_ue_spd})

            num_step += 1
            state = next_state

            elapsed_time = end_time - start_time
            # print("\r  Iteration {}: {} UES, {:.3f} s".format(
            #             iteration, len(env.prev_UEs), elapsed_time), end ='')

            # all_rewards += total_reward
            print("\r   Time: {}: {} UES, {:.3f} s, reward_upf_eff: {}, penalty_latency: {}, penalty_migration: {}, reward: {}"
                                    .format(iteration, len(env.prev_UEs), elapsed_time, penalty_num_upf, penalty_latency, penalty_migration, total_reward), end='')

            agent.tensorboard.update_stats(num_ues=len(env.prev_UEs), reward=total_reward, num_upf=penalty_num_upf, avg_latency=penalty_latency, migration_percentage=penalty_migration*100)

            if iteration >= batch_size and (not evaluate) and iteration < batch_size + 250:
                # batch_states = random.sample(states, batch_size)
                # batch_actions = random.sample(actions, batch_size)
                # batch_next_states = random.sample(next_states, batch_size)
                # batch_rewards = random.sample(rewards, batch_size)
                minibatch = random.sample(memory, batch_size)
                batch_states = [minibatch[m][0] for m in range(batch_size)]
                batch_actions = [minibatch[m][1] for m in range(batch_size)]
                batch_next_states = [minibatch[m][2] for m in range(batch_size)]
                batch_rewards = [minibatch[m][3] for m in range(batch_size)]

                loss = agent.learn(batch_states, batch_actions, batch_next_states, batch_rewards)
                agent.tensorboard.update_stats(loss=loss, epsilon=agent.epsilon/agent.epsilon_decay)
                agent.save(name)
            # if (iteration + 1) % batch_size == 0 and (not evaluate):
            #     # returns = agent.calc_returns(rewards)
            #     actor_loss, critic_loss = agent.learn(states, actions, next_states, rewards)
            #     agent.tensorboard.update_stats(actor_loss=actor_loss, critic_loss=critic_loss, epsilon=agent.epsilon/agent.epsilon_decay)
            #     agent.save(name)
            #     states = []
            #     actions = []
            #     next_states = []
            #     rewards = []

            # if total_reward > best_score:
            #     best_score = total_reward
            #     save_medel = True

            # if save_medel and (not evaluate):
            #     agent.save('upf_placement_' + name)
            
        # if total_reward > best_score:
        #    best_score = total_reward
        if not evaluate:
            agent.save(name)
