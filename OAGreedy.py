import numpy as np
from multiprocessing import Process, Queue, Barrier, Lock
from collections import deque
from NetworkEnv import NetworkEnv
from ModifiedTensorBoard import ModifiedTensorBoard
from sklearn.cluster import KMeans
import time


def get_deployment_overhead(BSs_with_UPF_previous, BSs_with_UPF, time_deployment, time_removal):
    intersection_size = len(BSs_with_UPF_previous & BSs_with_UPF)
    deployed_upfs = len(BSs_with_UPF) - intersection_size
    removed_upfs = len(BSs_with_UPF_previous) - intersection_size

    deployment_overhead = (deployed_upfs *
                           time_deployment + removed_upfs * time_removal)

    return deployment_overhead

def get_objective_function(G, f1_num_hops, f2_deployment_overhead, f3_control_plane_reassignment_overhead,
                           alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops):
    return alpha1 * get_f1_normalized(f1_num_hops, max_num_hops) + \
        alpha2 * get_f2_normalized(G, f2_deployment_overhead, time_deployment, time_removal) + \
        alpha3 * get_f3_normalized(f3_control_plane_reassignment_overhead,
                                   cost_relocation, num_UEs, max_num_hops)

def get_f1_normalized(f1_num_hops, max_num_hops):
    return f1_num_hops / max_num_hops


def get_f2_normalized(G, f2_deployment_overhead, time_deployment, time_removal):
    max_num_UPFs = G.number_of_nodes()
    max_deployment_overhead = max_num_UPFs * max(time_deployment, time_removal)
    return f2_deployment_overhead / max_deployment_overhead


def get_f3_normalized(f3_control_plane_reassignment_overhead, cost_relocation, num_UEs, max_num_hops):
    # EXPERIMENTAL
    max_control_plane_reassignment_overhead = (
        1 + num_UEs) * max_num_hops * cost_relocation
    return f3_control_plane_reassignment_overhead / max_control_plane_reassignment_overhead


if __name__ == '__main__':
  for num_UPFs in [4,5,6]:
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
    name = "oagreedy_{:d}upf".format(int(num_UPFs))    
    
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


    UE_to_UPF_assignment_previous = {}

    for iteration in range(800): #8640

        save_medel = True
        
        change_upf = 0

        # Transition Dynamics
        action = []
        BS_to_UPF_assignment = {}
        latencies_agg_dict = {}
        ues_agg_dict = {}
        tot_ues = 0
        for bs in env.G.nodes:
            tot_ues += bs.get_numUEs()

            for ue in bs.get_UEs():
                upf_node_previous = -1
                if ue.get_id() in UE_to_UPF_assignment_previous:
                    upf_node_previous = UE_to_UPF_assignment_previous[ue.get_id()]
                if bs.get_id() not in latencies_agg_dict:
                    latencies_agg_dict[bs.get_id()] = {}
                if upf_node_previous not in latencies_agg_dict[bs.get_id()]:
                    latencies_agg_dict[bs.get_id(
                    )][upf_node_previous] = env.max_num_hops + 1
                if bs.get_id() not in ues_agg_dict:
                    ues_agg_dict[bs.get_id()] = {}
                if upf_node_previous not in ues_agg_dict[bs.get_id()]:
                    ues_agg_dict[bs.get_id()][upf_node_previous] = []

                ues_agg_dict[bs.get_id()][upf_node_previous].append(ue.get_id())

        done_BSs = [False for _ in range(env.N)]

        for _ in range(num_UPFs):
            best_bs = None
            best_f_objective_function = None
            for bs in env.G.nodes:
                if not done_BSs[bs.get_id()]:
                    # Check objective function if bs is selected

                    # f2: Deployment overhead
                    f2_deployment_overhead = get_deployment_overhead(
                        set(old_action), set(action), 1, 0.1)

                    # f1 + f2: Vehicle latency (90-th percentile) and control-plane reassignment overhead
                    f_objective_function = get_objective_function(
                        env.G, 0, f2_deployment_overhead, 0, 0.5, 0.25, 0.25, 1, 0.1, 1, tot_ues, env.max_num_hops)
                    for bs2 in env.G.nodes:
                        if bs2.get_numUEs() == 0:
                            continue

                        for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                            num_UEs_agg = len(
                                ues_agg_dict[bs2.get_id()][upf_node_previous])
                            latency_agg = latencies_agg_dict[bs2.get_id(
                            )][upf_node_previous]
                            f3_control_plane_reassignment_overhead = 0
                            if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                                f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                    env.G_shortest_path_lengths[env.BSs[upf_node_previous]][env.BSs[BS_to_UPF_assignment[bs2.get_id(
                                    )][upf_node_previous]]] * 1

                            f_objective_keep = get_objective_function(env.G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                                      0.5, 0.25, 0.25, 1, 0.1, 1, tot_ues, env.max_num_hops)

                            f3_control_plane_reassignment_overhead = 0
                            if (upf_node_previous != -1):
                                f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                    env.G_shortest_path_lengths[env.BSs[upf_node_previous]
                                                            ][bs] * 1
                            f_objective_relocate = get_objective_function(
                                env.G, num_UEs_agg * env.G_shortest_path_lengths[bs2][bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, 0.5, 0.25, 0.25, 1, 0.1, 1, tot_ues, env.max_num_hops)

                            if f_objective_relocate < f_objective_keep:
                                f_objective_function += f_objective_relocate
                            else:
                                f_objective_function += f_objective_keep

                    if best_bs == None or f_objective_function < best_f_objective_function:
                        best_bs = bs
                        best_f_objective_function = f_objective_function

            previous_f_objective_function = best_f_objective_function

            upf_node = best_bs.get_id()
            done_BSs[upf_node] = True
            action.append(upf_node)

            for bs2 in env.G.nodes:
                if bs2.get_numUEs() == 0:
                    continue

                for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                    num_UEs_agg = len(
                        ues_agg_dict[bs2.get_id()][upf_node_previous])
                    latency_agg = latencies_agg_dict[bs2.get_id(
                    )][upf_node_previous]
                    f3_control_plane_reassignment_overhead = 0
                    if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                        f3_control_plane_reassignment_overhead = num_UEs_agg * \
                            env.G_shortest_path_lengths[env.BSs[upf_node_previous]][env.BSs[BS_to_UPF_assignment[bs2.get_id(
                            )][upf_node_previous]]] * 1

                    f_objective_keep = get_objective_function(env.G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                                0.5, 0.25, 0.25, 1, 0.1, 1, tot_ues, env.max_num_hops)

                    f3_control_plane_reassignment_overhead = 0
                    if (upf_node_previous != -1):
                        f3_control_plane_reassignment_overhead = num_UEs_agg * \
                            env.G_shortest_path_lengths[env.BSs[upf_node_previous]
                                                    ][best_bs] * 1

                    f_objective_relocate = get_objective_function(
                        env.G, num_UEs_agg * env.G_shortest_path_lengths[bs2][best_bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, 0.5, 0.25, 0.25, 1, 0.1, 1, tot_ues, env.max_num_hops)

                    if (bs2.get_id() not in BS_to_UPF_assignment or upf_node_previous not in BS_to_UPF_assignment[bs2.get_id()]) or f_objective_relocate < f_objective_keep:
                        new_latency = env.G_shortest_path_lengths[bs2][best_bs]
                        latencies_agg_dict[bs2.get_id()][upf_node_previous] = new_latency
                        # Update the assignment of all the UEs in bs2
                        if bs2.get_id() not in BS_to_UPF_assignment:
                            BS_to_UPF_assignment[bs2.get_id()] = {}
                        BS_to_UPF_assignment[bs2.get_id()][upf_node_previous] = upf_node

        # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment
        UE_to_UPF_assignment_previous = {}
        best_latency = 0
        for bs2_id in BS_to_UPF_assignment:
            for upf_node_previous in BS_to_UPF_assignment[bs2_id]:
                best_latency += len(ues_agg_dict[bs2_id][upf_node_previous]) * env.G_shortest_path_lengths[env.BSs[bs2_id]][env.BSs[BS_to_UPF_assignment[bs2_id][upf_node_previous]]]
                for ue_id in ues_agg_dict[bs2_id][upf_node_previous]:
                    UE_to_UPF_assignment_previous[ue_id] = BS_to_UPF_assignment[bs2_id][upf_node_previous]
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
