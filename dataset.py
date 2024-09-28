import math
import numpy as np
import random as rnd
import networkx as nx

PL_THRESHOLD = 2
DISTANCE_THRESHOLD = 500

class UE:
    def __init__(self, id, x=0, y=0, speed=0, bs=None, pred_x=0, pred_y=0, pred_bs=None):
        self._id = id
        self._x = float(x)
        self._y = float(y)
        self._speed = float(speed)
        self._bs = bs
        self._pl = float('inf')
        self._upf = -1
        self._pred_bs = pred_bs
        self._pred_x = pred_x
        self._pred_y = pred_y
        if self._bs is not None:
            self._bs.add_UE(self)
        if self._pred_bs is not None:
            self._pred_bs.add_pred_UE(self)
        self._prev_bs = -1

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y
    
    def get_speed(self):
        return self._speed

    def get_coords(self):
        return [self._x, self._y]

    def set_coords(self, x, y):
        self._x, self._y = (float(x), float(y))
    
    def set_speed(self, speed):
        self._speed = float(speed)

    def get_bs(self):
        return self._bs

    def set_bs(self, bs):
        if self._bs is not None:
            self._bs.remove_UE(self)
        if bs is not None:
            bs.add_UE(self)
        self._bs = bs

    def get_pred_coords(self):
        return [self._pred_x, self._pred_y]

    def set_pred_coords(self, pred_x, pred_y):
        self._pred_x, self._pred_y = (float(pred_x), float(pred_y))

    def get_pred_bs(self):
        return self._pred_bs

    def set_pred_bs(self, pred_bs):
        if self._pred_bs is not None:
            self._pred_bs.remove_pred_UE(self)
        if pred_bs is not None:
            pred_bs.add_pred_UE(self)
        self._pred_bs = pred_bs

    def get_id(self):
        return self._id

    def get_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            # distance = self._bs.get_distance_coords(self._x, self._y)
            # return compute_path_loss(distance)
            return self._pl
    
    def set_pl(self, pl):
        self._pl = pl

    def get_pred_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            distance = self._bs.get_distance_coords(self._pred_x, self._pred_y)
            return compute_path_loss(distance)

    def update_bs(self, bs, pl=float("inf")):
        if bs is None or pl + PL_THRESHOLD < self.get_pl():
            self.set_bs(bs)
            self.set_pl(pl)

    def update(self, x, y, speed, bs, pl=float("inf")):
        self.set_coords(x, y)
        self._speed = float(speed)
        self.update_bs(bs, pl)

    def update_unconditional(self, x, y, speed, bs):
        self.set_coords(x, y)
        self._speed = float(speed)
        self.set_bs(bs)

    def update_pred_bs(self, pred_bs, pl=float("inf")):
        if pred_bs is None or pl + PL_THRESHOLD < self.get_pred_pl():
            self.set_pred_bs(pred_bs)

    def update_pred(self, pred_x, pred_y, pred_bs, pl=float("inf")):
        self.set_pred_coords(pred_x, pred_y)
        self.update_pred_bs(pred_bs, pl)

    def update_pred_unconditional(self, pred_x, pred_y, pred_bs):
        self.set_pred_coords(pred_x, pred_y)
        self.set_pred_bs(pred_bs)


class BS:
    def __init__(self, id_, x, y, UPF=False):
        self._id = int(id_)
        self._x = float(x)
        self._y = float(y)
        self._UPF = UPF
        self.UEs = []
        self.pred_UEs = []

    def get_id(self):
        return self._id

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_coords(self):
        return [self._x, self._y]

    def add_UE(self, ue):
        self.UEs.append(ue)

    def remove_UE(self, ue):
        self.UEs.remove(ue)

    def get_UEs(self):
        return self.UEs

    def get_numUEs(self):
        return len(self.UEs)

    def clear_UEs(self):
        self.UEs = []

    def add_pred_UE(self, ue):
        self.pred_UEs.append(ue)

    def remove_pred_UE(self, ue):
        self.pred_UEs.remove(ue)

    def get_pred_UEs(self):
        return self.pred_UEs

    def get_pred_numUEs(self):
        return len(self.pred_UEs)

    def clear_pred_UEs(self):
        self.pred_UEs = []

    def get_distance(self, bs2):
        return ((bs2.get_x()-self.get_x())**2 + (bs2.get_y()-self.get_y())**2)**0.5

    def get_distance_coords(self, x, y):
        return ((x-self.get_x())**2 + (y-self.get_y())**2)**0.5


''' Generates a set of connected components by conecting all the base stations
    that are positioned less than DISTANCE_THRESHOLD meters apart
'''
def generate_graph(bs_file):
    G = nx.Graph()
    BSs = {}
    highest_bs_id = -1

    with open(bs_file) as f:
        for _, line in enumerate(f):
            bs_data = line.strip().split()

            bs = BS(bs_data[0], bs_data[1], bs_data[2])

            BSs[bs.get_id()] = bs
            if bs.get_id() > highest_bs_id:
                highest_bs_id = bs.get_id()
            G.add_node(bs)
            for other_bs in G.nodes:
                dist = bs.get_distance(other_bs)
                if other_bs is not bs and dist < DISTANCE_THRESHOLD:
                    G.add_edge(bs, other_bs, delay=dist*0.005)

    join_components(G)

    #G_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    G_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='delay'))

    max_num_hops = nx.algorithms.distance_measures.diameter(G)

    return G, BSs, G_shortest_path_lengths, highest_bs_id, max_num_hops


''' Connects the connected components to compose a single giant component.
    This procedure is performed as follow: On each iteration, from the second
    biggest connected component, the node which is at the shortest distance
    from one of the nodes of the giant component is determined, and an edge
    between these two nodes.

    This way, on each iteration the connected components are joined to the giant
    component in order of size to achieve a single giant component.
'''


def join_components(G):
    while True:
        connected_components = list(nx.connected_components(G))
        if len(list(connected_components)) < 2:
            break
        connected_components = sorted(
            connected_components, key=len, reverse=True)

        bs1 = bs2 = distance = None
        for bs in connected_components[1]:
            for bs_giant_component in connected_components[0]:
                d = bs.get_distance(bs_giant_component)
                if distance is None or d < distance:
                    distance = d
                    bs1 = bs
                    bs2 = bs_giant_component
        G.add_edge(bs1, bs2, delay=distance*0.005)


''' Determines the nearest BS to the coordinates x and y
'''


def get_optimal_bs(BSs, x, y):
    distance = None
    bs = None
    for node in BSs.values():
        d = node.get_distance_coords(x, y)
        if distance is None or d < distance:
            distance = d
            bs = node
    return bs, distance


def compute_path_loss(distance):
    # p1 = 46.61
    # p2 = 3.63
    # std = 9.83
    # return p1 + (p2 * 10 * log10(distance)) + gauss(0, std)
    return 46.61 + (3.63 * 10 * math.log10(distance)) + rnd.gauss(0, 9.83)


def get_minimum_hops_from_BS_to_any_UPF(G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths):
    hops = None
    bs_with_upf = None
    for other_bs_id in BSs_with_UPF_ids:
        other_bs = BSs[other_bs_id]
        try:
            # h = len(nx.shortest_path(G, source=bs,
            #                          target=other_bs)) - 1  # Dijkstra
            # Pre-computed Floyd-Wharsall
            h = G_shortest_path_lengths[bs][other_bs]
        except:
            continue
        if hops is None or h < hops:
            hops = h
            bs_with_upf = other_bs

    if hops is None:
        # To handle the cases in which there is not an UPF deployed
        hops = G.number_of_nodes() / 2
        # raise Exception("No reachable UPF from BS {}".format(bs.get_id()))

    return hops, bs_with_upf


''' Generates the list of UEs of the new iteration: updates data from previous
    iteration, removes UEs that do not appear in the new stage and adds those
    that did not appear in the last stage.
'''
def read_UE_data(ue_file, BSs, iteration_duration, num_iterration = -1):
    UEs_last_iteration = {}

    first_timestamp_iteration = None

    UEs_new_iteration = {}

    iter = 0
    with open(ue_file) as f, open("koln_with_bs.tr", 'a') as f2:
        for line in f:
            # Read UEs from new iteration
            line2 = line
            line = line.strip().split()
            
            timestamp = int(line[0])
            id_ = int(line[1].split("_")[0].split("#")[0])
            x = float(line[2])
            y = float(line[3])
            speed = float(line[4])  # Unused
            pl = None
            if len(line) > 5:
                bs = BSs[int(line[5])]
                # This line needs to be uncommented in order to have histeresis
                pl = float(line[6])
            else:
                bs, distance = get_optimal_bs(BSs, x, y)
                pl = compute_path_loss(distance)
                # line2 = line2.strip() + ' {} {}\n'.format(bs.get_id(), pl)
                # f2.writelines(line2)

            pred_x = float(line[2])
            pred_y = float(line[3])

            if len(line) > 7:
                pred_x = float(line[6])
                pred_y = float(line[7])
                # pred_x = float(line[2])
                # pred_y = float(line[3])

            pred_pl = None
            if len(line) > 8:
                pred_bs = BSs[int(line[8])]
                # This line needs to be uncommented in order to have histeresis
                #pred_pl = compute_path_loss(pred_bs.get_distance_coords(pred_x, pred_y))

            else:
                pred_bs, pred_distance = get_optimal_bs(BSs, pred_x, pred_y)
                pred_pl = compute_path_loss(pred_distance)

            if first_timestamp_iteration == None:
                first_timestamp_iteration = timestamp

            if timestamp - first_timestamp_iteration > iteration_duration:
                # Iteration finished: Yield results
                for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
                    ue.update_bs(None)
                    ue.update_pred_bs(None)

                UEs_last_iteration = UEs_new_iteration
                yield first_timestamp_iteration, UEs_new_iteration

                iter += 1
                if iter >= num_iterration and num_iterration > 0:
                    return 

                # Resumed execution for next iteration: Initialize values for this iteration
                UEs_new_iteration = {}
                first_timestamp_iteration = timestamp

            # Update UE already present in previous iteration
            if id_ in UEs_last_iteration:
                ue = UEs_last_iteration[id_]
                if pl:
                    ue.update(x, y, speed, bs, pl)
                else:
                    ue.update_unconditional(x, y, speed, bs)
                if pred_pl:
                    ue.update_pred(pred_x, pred_y, pred_bs, pred_pl)
                    #ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
                else:
                    ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
            # Only the last appearance of each UE in the iteration is considered
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                if pl:
                    ue.update(x, y, speed, bs, pl)
                else:
                    ue.update_unconditional(x, y, speed, bs)
                if pred_pl:
                    ue.update_pred(pred_x, pred_y, pred_bs, pred_pl)
                    #ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
                else:
                    ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
            # Se crea un nuevo UE
            else:
                ue = UE(id_, x, y, speed, bs, pred_x, pred_y, pred_bs)
            UEs_new_iteration[id_] = ue


if __name__ == "__main__":
    import time
    save_file = open('pre_state.txt', 'a')
    ue_file = "koln.tr"
    bs_file = "koln_bs-deployment-D1_fixed.log"
    G, BSs, G_shortest_path_lengths, highest_bs_id, max_num_hops = generate_graph(
            bs_file)
    
    # bs_x = []
    # bs_y = []
    # for bs in G.nodes():
    #     bs_x.append(bs.get_x())
    #     bs_y.append(bs.get_y())

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(bs_x, bs_y, 'o')
    # plt.grid()
    # plt.show()
    iteration = 0
    start_time = time.process_time()
    for UEs in read_UE_data(ue_file, BSs, 5, -1):
        end_time = time.process_time()
        iteration += 1
        print(iteration, len(UEs[1]),end_time - start_time)

        # state_lowSpeedUser = [0] * len(BSs)
        # state_highSpeedUser = [0] * len(BSs)

        # for bs in G.nodes():
        #     ue_to_bs = bs.get_UEs()
        #     for ue in bs.get_UEs():
        #         spd = ue.get_speed()
        #         if spd < 20:
        #             state_lowSpeedUser[bs.get_id()] += 1
        #         else:
        #             state_highSpeedUser[bs.get_id()] += 1
        # print(UEs[0], state_lowSpeedUser, state_highSpeedUser, file=save_file)
        start_time = time.process_time()