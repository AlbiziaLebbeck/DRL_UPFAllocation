import numpy as np
from multiprocessing import Process, Queue, Barrier, Lock
from collections import deque
from NetworkEnv import NetworkEnv
from ModifiedTensorBoard import ModifiedTensorBoard
from sklearn.cluster import KMeans
import time


if __name__ == '__main__':
    ue_file = "koln.tr"#args.ueFile
    bs_file = "koln_bs-deployment-D1_fixed.log"#args.bsFile
    episodes = 50
    batch_size = 100
    evaluate = True

    # global_env = NetworkEnv(bs_file, ue_file)
    # global_agent = ACAgent(global_env)

    best_score = -float('inf')

    # Create Environment and set number of speed class for vehicle here
    env = NetworkEnv(bs_file, ue_file, 1)
    
    a1 = 0
    a2 = 1
    a3 = 0

    # name = "new_a2c5nodes-{}".format(int(time.time()))
    name = "kmeangreedy_3upf"    
    
    tensorboard = ModifiedTensorBoard("ac5nodes", log_dir="logs_{}nodes/{}".format(env.N, name))

    # if e % 10 == 0:
    #     all_states = []
    #     all_actions = []
    #     all_returns = []
    
    # states = deque(maxlen=1000)
    # actions = deque(maxlen=1000)
    # next_states = deque(maxlen=1000)
    # rewards = deque(maxlen=1000)
    memory = deque(maxlen=1000)


    num_step = 0
    
    state = env.reset()
    old_action = []
    for iteration in range(800): #8640

        save_medel = True
        
        change_upf = 0

        # Transition Dynamics
        action = []
        features = []
        for bs in env.G.nodes:
          if bs.get_numUEs() > 0:
            features.append([bs.get_x(), bs.get_y()])
        
        num_UPFs = 3
        kmeans = KMeans(
          init="k-means++",  # "random" / "k-means++"
          n_clusters=min(num_UPFs, len(features)),
          n_init=2,
          max_iter=1000,
          random_state=0  # To allow for reproducibility
        )
        
        kmeans.fit(features)
        
        cluster_BS_ids_list = [[] for _ in range(num_UPFs)]
        for i in range(len(kmeans.labels_)):
          cluster_BS_ids_list[kmeans.labels_[i]].append(i)
          
        tot_ues = 0
        for bs in env.G.nodes:
          tot_ues += bs.get_numUEs()
        
        best_latency = 0
        for cluster in range(num_UPFs):
          cluster_BS_ids = cluster_BS_ids_list[cluster]
          
          best_bs = None
          best_acc_latency = None
          for bs_index in cluster_BS_ids:
            bs = env.BSs[bs_index]
            # Check latency if bs is selected
            acc_latency = 0
            for bs2_index in cluster_BS_ids:
              bs2 = env.BSs[bs2_index]
              new_latency = env.G_shortest_path_lengths[bs2][bs]
              acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
              best_bs = bs
              best_acc_latency = acc_latency

          if best_bs != None:
            best_latency += best_acc_latency
            action.append(best_bs.get_id())

        best_latency /= tot_ues 

        upf_placement = [0] * env.N
        for a in action:
          upf_placement[a] = 1
          if not(a in old_action):
            change_upf += 1
                
        old_action = action
        
        start_time = time.process_time()
        next_state, reward, done, timestamp = env.step(upf_placement)
        end_time = time.process_time()

        if done:
            break 
        
        tensorboard.step = iteration + 1
        
        penalty_num_upf = reward[0] 
        penalty_latency = reward[1]
        penalty_migration = reward[2]
        total_reward = - a1 * penalty_num_upf/env.N - a2*best_latency - a3*penalty_migration / (len(env.prev_UEs) * env.max_num_hops)

        # if not done:
        # states.append(state)
        # actions.append(action)
        # next_states.append(next_state)
        # rewards.append(total_reward)
        memory.append([state,action,next_state,total_reward])

        for spd in range(1):
            num_ue_spd = sum(state[spd*env.N:(spd+1)*env.N])
            tensorboard.update_stats(**{"numUEs_s"+str(spd):num_ue_spd})

        num_step += 1
        state = next_state

        elapsed_time = end_time - start_time
        # print("\r  Iteration {}: {} UES, {:.3f} s".format(
        #             iteration, len(env.prev_UEs), elapsed_time), end ='')

        # all_rewards += total_reward
        print("\r   Time: {}: {} UES, {:.3f} s, reward_upf_eff: {}, penalty_latency: {}, penalty_migration: {}, reward: {}"
                                .format(iteration, len(env.prev_UEs), elapsed_time, penalty_num_upf, best_latency, penalty_migration, total_reward), end='')

        tensorboard.update_stats(num_ues=len(env.prev_UEs), reward=total_reward, num_upf=penalty_num_upf, change_upfs=change_upf, avg_latency=penalty_latency, migration_percentage=penalty_migration/len(env.prev_UEs))
