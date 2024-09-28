from dataset import *
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import time

class NetworkEnv(py_environment.PyEnvironment):

  def __init__(self, bs_file, ue_file, num_class):
    self.bs_file = bs_file
    self.ue_file = ue_file
    self.G, self.BSs,  self.G_shortest_path_lengths,  self.highest_bs_id,  self.max_num_hops = generate_graph(
        bs_file)
    self.gen_UE = read_UE_data(ue_file,  self.BSs, 5)
    self.N = len(self.BSs)
    self.num_class = num_class
    
    self.bsid_to_idx = {}
    for idx,bs in enumerate(self.BSs):
      self.bsid_to_idx[bs] = idx

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(self.N,), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=((num_class+1)*self.N,), dtype=np.int32, minimum=0, name='observation')
    
    self._state_User = [0] * self.N * self.num_class
    self._state_UPFDeployment = [0] * self.N
    self._state = self._state_User + self._state_UPFDeployment
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state_User = [0] * self.N * self.num_class
    self._state_UPFDeployment = [0] * self.N
    self._episode_ended = False

    self.G, self.BSs,  self.G_shortest_path_lengths,  self.highest_bs_id,  self.max_num_hops = generate_graph(
        self.bs_file)
    self.gen_UE = read_UE_data(self.ue_file,  self.BSs, 5)
    timestamp,UEs = next(self.gen_UE)

    for uid,ue in UEs.items():
      spd = ue.get_speed()
      bs = ue.get_bs()._id
      spd_class = int(spd // (60 / self.num_class))
      spd_class = spd_class if spd_class < self.num_class else self.num_class - 1
      self._state_User[self.bsid_to_idx[bs] + spd_class * self.N] += 1

    self.prev_UEs = UEs
    self._state_User = [u/sum(self._state_User) for u in self._state_User]

    self._state = self._state_User + self._state_UPFDeployment
    return self._state

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
    
    self._state_User = [0] * self.N * self.num_class

    list_latency = []

    penalty_num_upf = 0
    penalty_latency = 0
    penalty_migration = 0

    #convert to list of binary
    # action = [int(digit) for digit in bin(action+1)[2:]]
    # action = (self.N-len(action))*[0] + action
    # action = [1 if i==action else 0 for i in range(self.N)]

    #Set Deployment for the first 10 nodes 
    BSs_with_UPF_ids = [list(self.BSs.keys())[i] for i in range(self.N) if action[i] == 1]
    
    try:
      
      start_time = time.process_time()

      timestamp,UEs = next(self.gen_UE)
      # print(timestamp)
      end_time = time.process_time()
      # print(end_time - start_time)
      
      # Penalty for cost of UPF deployment
      if len(BSs_with_UPF_ids) > 0:
        penalty_num_upf = len(BSs_with_UPF_ids)
      
      if len(BSs_with_UPF_ids) > 0:
        for bs in self.G.nodes():
          ue_to_bs = bs.get_UEs()
          if len(ue_to_bs) == 0:
            continue
          
          latency, upf_node = get_minimum_hops_from_BS_to_any_UPF(
              self.G,  self.BSs, bs, BSs_with_UPF_ids,  self.G_shortest_path_lengths)
          # Penalty for latency
          # if latency > self.max_num_hops//2:
          list_latency += len(ue_to_bs) * [latency] 
          # penalty_latency += 1 * len(ue_to_bs) * latency /  self.max_num_hops #/ len(UEs)
          for ue in bs.get_UEs():
              # Determine which node will have a UPF to serve this UE
              if ue._upf > 0:
                hops = self.G_shortest_path_lengths[self.BSs[ue._upf]][upf_node]
                penalty_migration += hops

              ue._upf = upf_node.get_id()
              
              spd = ue.get_speed()
              spd_class = int(spd // (60 / self.num_class))
              spd_class = spd_class if spd_class < self.num_class else self.num_class - 1
              self._state_User[self.bsid_to_idx[bs.get_id()] + spd_class * self.N] += 1

        penalty_latency = np.mean(list_latency)
      
      # if penalty_latency < -1:
      #   print(BSs_with_UPF_ids, len(UEs))

      self.prev_UEs = UEs
      self._state_User = [u/sum(self._state_User) for u in self._state_User]
      
      self._state_UPFDeployment = action
      self._state = self._state_User + self._state_UPFDeployment
    
    except StopIteration:
      self._episode_ended = True
      timestamp = 9999999

    reward = (penalty_num_upf, penalty_latency, penalty_migration)
    # print(len(self.prev_UEs), reward[0], reward[1], reward[2], sum(reward), file=open('reward.log', 'a'))
    
    return self._state, reward, self._episode_ended, timestamp
  
# env = NetworkEnv(bs_file)
# state = env.reset()
# state = env.step(372)