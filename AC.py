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


class ACAgent:
    def __init__(self, env, num_upf):
        self.state_size = env.observation_spec().shape
        self.action_size = env.action_spec().shape[0]
        self.num_upf = num_upf

        # Hyperparameters
        self.gamma = 0.99            # Discount rate
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.epsilon = 0
        self.epsilon_decay = 0.9
        self.min_epsilon = 0.0
        
        # Construct Actor and Critic models
        self.Actor, self.Critic = self._build_model()

    # Constructs NN
    def _build_model(self):
        input_state = Input(self.state_size)
        d1 = Dense(1024, activation='relu')(input_state)
        d2 = Dense(512, activation='relu')(d1)
        actions = [Dense(2, activation='softmax')(d2) for _ in range(self.action_size)]
        value = Dense(1)(d2)

        actors = []
        for i in range(self.action_size):
            actor = Model(inputs=input_state, outputs=actions[i])
            actor.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.actor_lr))
            actors.append(actor)

        critic = Model(inputs=input_state, outputs=value)
        critic.compile(loss='mse', optimizer=Adam(learning_rate=self.critic_lr))

        return actors, critic

    def action(self, state):
        actions = []
        
        for i in range(self.action_size):
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(2)
            else:
                action_probs = self.Actor[i](np.array([state]))
                action = np.random.choice(2, p=action_probs.numpy()[0])
            actions.append(action)
        
        return actions
    
    def calc_returns(self, rewards):
        returns = []
        sum_discounted_reward = 0
        rewards.reverse()
        for r in rewards:
           sum_discounted_reward = r + self.gamma*sum_discounted_reward
           returns.append(sum_discounted_reward)
        rewards.reverse()
        returns.reverse()
        return returns

    def learn(self, states, actions, next_states, rewards):

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        
        num_data = len(states)

        values = self.Critic(states)[:, 0]
        next_values = self.Critic(next_states)[:, 0]
        targets = rewards + self.gamma * next_values
        advantages = targets - values

        actor_loss = 0
        for i in range(self.action_size):
            onehot_action = np.zeros((num_data, 2), dtype=int)
            onehot_action[np.arange(num_data), actions[:,i]] = 1

            actors_prob = np.sum(self.Actor[i](states) * onehot_action, axis=1)
            importance_w = actors_prob/(self.epsilon/2+(1-self.epsilon)*actors_prob)
            importance_w = np.nan_to_num(importance_w, nan=1)

            actor_hist = self.Actor[i].fit(states, onehot_action, sample_weight=importance_w * advantages, epochs=1, verbose=0)
            actor_loss += actor_hist.history['loss'][0]
        
        critic_hist = self.Critic.fit(states, targets, epochs=1, verbose=0)
        critic_loss = critic_hist.history['loss'][0]

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon

        return actor_loss, critic_loss

    def load(self, name):
        for i in range(self.num_upf):
            self.Actor[i] = load_model('models/' + name + '_Actor'+str(i)+'.h5', compile=False)
        self.Critic = load_model('models/' + name + '_Critic.h5', compile=False)

    def save(self, name):
        for i in range(self.num_upf):
            self.Actor[i].save('models/' + name + '_Actor'+str(i)+'.h5')
        self.Critic.save('models/' + name + '_Critic.h5')


if __name__ == '__main__':
    for speeds in [1]:
        ue_file = "koln.tr"#args.ueFile
        bs_file = "koln_bs-deployment-D1_fixed.log"#args.bsFile
        episodes = 50
        batch_size = 100
        evaluate = False

        # global_env = NetworkEnv(bs_file, ue_file)
        # global_agent = ACAgent(global_env)

        best_score = -float('inf')

        # Create Environment and set number of speed class for vehicle here
        env = NetworkEnv(bs_file, ue_file, speeds)
        
        a1 = 0
        a2 = 1
        a3 = 0

        # name = "new_a2c5nodes-{}".format(int(time.time()))
        name = "ac_allupf_{:d}spd_{:03d}_{:03d}_1".format(speeds, int(a2*100), int(a3*100))
        
        agent = ACAgent(env, 1)
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
            upf_placement = action
            
            start_time = time.process_time()
            next_state, reward, done, timestamp = env.step(upf_placement)
            end_time = time.process_time()

            if done:
                break 
            
            agent.tensorboard.step = iteration + 1
            
            penalty_num_upf = reward[0] 
            penalty_latency = reward[1]
            penalty_migration = reward[2]
            total_reward = - a1 * penalty_num_upf/env.N - a2*penalty_latency - a3*penalty_migration / (len(env.prev_UEs) * env.max_num_hops)

            # if not done:
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

            agent.tensorboard.update_stats(num_ues=len(env.prev_UEs), reward=total_reward, num_upf=penalty_num_upf, avg_latency=penalty_latency, migration_percentage=penalty_migration/len(env.prev_UEs))

            if iteration >= batch_size and (not evaluate) and iteration < batch_size+250:
                minibatch = random.sample(memory, batch_size)
                batch_states = [minibatch[m][0] for m in range(batch_size)]
                batch_actions = [minibatch[m][1] for m in range(batch_size)]
                batch_next_states = [minibatch[m][2] for m in range(batch_size)]
                batch_rewards = [minibatch[m][3] for m in range(batch_size)]
                actor_loss, critic_loss = agent.learn(batch_states, batch_actions, batch_next_states, batch_rewards)
                agent.tensorboard.update_stats(actor_loss=actor_loss, critic_loss=critic_loss, epsilon=agent.epsilon/agent.epsilon_decay)
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