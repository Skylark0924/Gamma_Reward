# from utils.attention import AttentionCritic
import copy
import json
import math
import operator
import os
import sys
import time

import cityflow
import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils.utility import parse_roadnet

gym.logger.set_level(40)


class CityFlowEnvRay(MultiAgentEnv):
    """
    multi intersection cityflow environment, for the Ray framework
    """
    observation_space = Box(0.0 * np.ones((29,)), 150 * np.ones((29,)))
    action_space = Discrete(8)  # num of agents
    config = json.load(open('/home/skylark/PycharmRemote/Gamma-Reward-Perfect/config/global_config.json'))
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    intersection_id = list(config['lane_phase_info'].keys())

    def __init__(self, env_config):
        config = json.load(open('/home/skylark/PycharmRemote/Gamma-Reward-Perfect/config/global_config.json'))
        cityflow_config = json.load(open(config['cityflow_config_file']))
        self.roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
        self.record_dir = '/home/skylark/PycharmRemote/Gamma-Reward-Perfect/record/' + env_config["Name"]
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

        self.dic_traffic_env_conf = {
            'ADJACENCY_BY_CONNECTION_OR_GEO': False,
            'TOP_K_ADJACENCY': 5
        }

        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}
        self.dic_lane_vehicle_previous_step = {}
        self.traffic_light_node_dict = self._adjacency_extraction()
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}
        self.list_lane_vehicle_previous_step = {}
        self.list_lane_vehicle_current_step = {}
        self.dic_vehicle_arrive_leave_time = {}

        config["lane_phase_info"] = parse_roadnet(self.roadnetFile)
        intersection_id = list(config['lane_phase_info'].keys())
        self.Gamma_Reward = env_config["Gamma_Reward"]
        self.threshold = env_config["threshold"]
        self.min_action_time = env_config["MIN_ACTION_TIME"]
        self.road_sort = env_config['road_sort']

        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config["thread_num"])
        # self.eng = config["eng"][0]
        self.num_step = 3600
        self.intersection_id = intersection_id  # list, [intersection_id, ...]
        self.num_agents = len(self.intersection_id)
        self.state_size = None
        self.lane_phase_info = config["lane_phase_info"]  # "intersection_1_1"

        # self.score = []
        # self.score_file = './utils/score_' + str(datetime.datetime.now()) + '.csv'
        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {}  # {id_:[lanes]}
        self.empty = {}
        self.dic_num_id_inter = {}

        num_id = 0
        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0

            self.dic_lane_vehicle_current_step[id_] = {}
            self.dic_lane_waiting_vehicle_count_current_step[id_] = {}
            self.dic_vehicle_arrive_leave_time[id_] = {}
            self.empty[id_] = {}
            self.dic_num_id_inter[num_id] = id_
            num_id += 1

        self.reset_count = 0
        self.get_state()  # set self.state_size
        self.num_actions = len(self.phase_list[self.intersection_id[0]])

        self.count = 0
        self.congestion_count = 0
        # self.done = False
        self.congestion = False
        self.iteration_count = []

        self.reset()

    def reset(self):
        print("\033[1;34m=================================\033[0m")
        print("\033[1;34mreset_count: {0}, iteration: {1}\033[0m".format(self.reset_count, self.count))
        # self.iteration_count.append(self.count)
        # if self.reset_count >= 102:
        #     df = pd.DataFrame(self.iteration_count)
        #     df.to_csv(os.path.join(self.record_dir, 'iteration_count.csv'))

        if not operator.eq(self.dic_vehicle_arrive_leave_time, self.empty):
            path_to_log = self.record_dir + '/train_results/episode_{0}/'.format(self.reset_count)
            if not os.path.exists(path_to_log):
                os.makedirs(path_to_log)
            self.log(path_to_log)
            print("Log is saved !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(path_to_log)

        self.eng.reset()
        self.dic_vehicle_arrive_leave_time = copy.deepcopy(self.empty)
        self.dic_lane_vehicle_current_step = copy.deepcopy(self.empty)
        self.dic_lane_waiting_vehicle_count_current_step = copy.deepcopy(self.empty)
        # self.traffic_light_node_dict = self._adjacency_extraction()
        # self.done = False
        self.congestion = False
        self.count = 0
        self.congestion_count = 0
        self.reset_count += 1
        return {id_: np.zeros((self.state_size,)) for id_ in self.intersection_id}

    def step(self, action):
        """
        Calculate the state in time
        """
        step_start_time = time.time()
        for i in range(self.min_action_time):
            self._inner_step(action)

        state = self.get_state()
        reward = self.get_raw_reward()

        # 判断是否已经出现拥堵
        self.congestion = self.compute_congestion()
        self.done = {id_: False for id_ in self.intersection_id}
        self.done['__all__'] = False
        # if self.count >= self.num_step:
        #     self.done = {id_: True for id_ in self.intersection_id}
        #     self.done['__all__'] = True
        # if self.count == 3600:
        #     self.reset()
        return state, reward, self.done, {}

    def _inner_step(self, action):
        self.update_previous_measurements()

        for id_, a in action.items():  # intersection_id, corresponding action
            if self.current_phase[id_] == self.phase_list[id_][a]:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = self.phase_list[id_][a]
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])  # set phase of traffic light

        self.eng.next_step()
        self.count += 1

        # print(self.count)

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": None,
                              "get_vehicle_distance": None
                              }
        for id_ in self.intersection_id:
            self.update_current_measurements_map(id_, self.system_states)

    def update_previous_measurements(self):
        self.dic_lane_vehicle_previous_step = copy.deepcopy(self.dic_lane_vehicle_current_step)
        self.dic_lane_waiting_vehicle_count_previous_step = copy.deepcopy(
            self.dic_lane_waiting_vehicle_count_current_step)
        self.dic_vehicle_speed_previous_step = copy.deepcopy(self.dic_vehicle_speed_current_step)
        self.dic_vehicle_distance_previous_step = copy.deepcopy(self.dic_vehicle_distance_current_step)

    def update_current_measurements_map(self, id_, simulator_state):
        ## need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        for lane in self.lane_phase_info[id_]['start_lane']:
            self.dic_lane_vehicle_current_step[id_][lane] = simulator_state["get_lane_vehicles"][lane]

            self.dic_lane_waiting_vehicle_count_current_step[id_][lane] = \
                simulator_state["get_lane_waiting_vehicle_count"][
                    lane]

        for lane in self.lane_phase_info[id_]['end_lane']:
            self.dic_lane_vehicle_current_step[id_][lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[id_][lane] = \
                simulator_state["get_lane_waiting_vehicle_count"][
                    lane]

        self.dic_vehicle_speed_current_step[id_] = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step[id_] = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step[id_] = _change_lane_vehicle_dic_to_list(
            self.dic_lane_vehicle_current_step[id_])
        self.list_lane_vehicle_previous_step[id_] = _change_lane_vehicle_dic_to_list(
            self.dic_lane_vehicle_previous_step[id_])

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step[id_]) - set(self.list_lane_vehicle_previous_step[id_]))
        # if id_ == 'intersection_6_1':
        #     print('list_lane_vehicle_current_step: ' + str(self.list_lane_vehicle_current_step))
        #     print('list_lane_vehicle_previous_step: ' + str(self.list_lane_vehicle_previous_step))

        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step[id_]) - set(self.list_lane_vehicle_current_step[id_]))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle(id_)
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l
        # print('list_vehicle_new_arrive' + str(list_vehicle_new_arrive))
        # print('list_vehicle_new_left_entering_lane' + str(list_vehicle_new_left_entering_lane))

        # update vehicle arrive and left time
        self._update_arrive_time(id_, list_vehicle_new_arrive)
        self._update_left_time(id_, list_vehicle_new_left_entering_lane)

        # update vehicle minimum speed in history, # to be implemented
        # self._update_vehicle_min_speed()

        # update feature
        # self._update_feature_map(simulator_state)

    def compute_congestion(self):
        index = False
        intersection_info = {}
        for id_ in self.intersection_id:
            intersection_info[id_] = self.intersection_info(id_)
        congestion = {id_: False for id_ in self.intersection_id}
        for id_ in self.intersection_id:
            if np.max(list(intersection_info[id_]["start_lane_waiting_vehicle_count"].values())) > self.threshold:
                congestion[id_] = True
                index = True
        return index

    def get_state(self):
        state = {id_: self._get_state(id_) for id_ in self.intersection_id}
        # self.score.append(self.get_score())
        # if self.reset_count >= 100:
        #     '''
        #     TODO: save fn may be a bug
        #     '''
        # score = pd.DataFrame(self.score)
        # score.to_csv(self.score_file)
        return state

    def _get_state(self, id_):
        state = self.intersection_info(id_)

        if self.Gamma_Reward:
            #### dw ####
            keys = state['end_lane_vehicle_count'].keys()
            start_index = id_.find('_')
            s0 = 'road' + id_[start_index: start_index + 4] + '_0'  # To East
            s1 = 'road' + id_[start_index: start_index + 4] + '_1'  # To North
            s2 = 'road' + id_[start_index: start_index + 4] + '_2'  # To West
            s3 = 'road' + id_[start_index: start_index + 4] + '_3'  # To South

            num_w_e = 0
            num_e_w = 0
            num_n_s = 0
            num_s_n = 0

            for i in keys:
                if i.startswith(s0):
                    num_w_e += state['end_lane_vehicle_count'][i]
                elif i.startswith(s1):
                    num_n_s += state['end_lane_vehicle_count'][i]
                elif i.startswith(s2):
                    num_e_w += state['end_lane_vehicle_count'][i]
                elif i.startswith(s3):
                    num_s_n += state['end_lane_vehicle_count'][i]

            end_lane_dict = {s0: num_w_e, s1: num_n_s, s2: num_e_w, s3: num_s_n}
            end_lane_sorted_keys = sorted(end_lane_dict.keys())

            state_dict_waiting = state['start_lane_waiting_vehicle_count']
            state_dict = state['start_lane_vehicle_count']

            # 12-dim start lanes car + 12-dim start lanes waiting car number + current phase + 4-dim end_lanes
            # waiting car number
            return_state = [state_dict[key] for key in self.road_sort[id_]] + [state_dict_waiting[key]
                                                                               for key in self.road_sort[id_]] + [
                               state['current_phase']] + [end_lane_dict[key] for key in end_lane_sorted_keys]
            # return_state = [state_dict[key] for key in sorted_keys] + [state['current_phase']] + \
            #                [0, 0, 0, 0]
        else:
            state_dict = state['start_lane_waiting_vehicle_count']
            return_state = [state_dict[key] for key in self.road_sort[id_]] + [0] * 12 + [state['current_phase']] + \
                           [0, 0, 0, 0]

        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        """
        info of intersection 'id_'
        """
        state = {}

        get_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        get_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        # get_lane_vehicles = self.eng.get_lane_vehicles()
        # get_vehicle_speed = self.eng.get_vehicle_speed()

        state['start_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.end_lane[id_]}

        # state['lane_vehicle_count'] = state['start_lane_vehicle_count'].copy()
        # state['lane_vehicle_count'].update(state['end_lane_vehicle_count'].items())
        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in
                                                     self.start_lane[id_]}
        # state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in
        #                                            self.end_lane[id_]}
        #
        # state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        # state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}
        #
        # state['start_lane_speed'] = {
        #     lane: np.sum(list(map(lambda vehicle: get_vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (
        #             get_lane_vehicle_count[lane] + 1e-5) for lane in
        #     self.start_lane[id_]}  # compute start lane mean speed
        # state['end_lane_speed'] = {
        #     lane: np.sum(list(map(lambda vehicle: get_vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (
        #             get_lane_vehicle_count[lane] + 1e-5) for lane in
        #     self.end_lane[id_]}  # compute end lane mean speed

        state['current_phase'] = self.current_phase[id_]
        # state['current_phase_time'] = self.current_phase_time[id_]

        state['adjacency_matrix'] = self.traffic_light_node_dict[id_]['adjacency_row']

        return state

    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(np.array(return_state), [1, self.state_size]).flatten()
        return return_state

    def get_raw_reward(self):
        reward = {id_: self._get_raw_reward(id_) for id_ in self.intersection_id}
        # mean_global_sum = np.mean(list(reward.values()))

        return reward

    def _get_raw_reward(self, id_):
        # every agent/intersection's reward
        state = self.intersection_info(id_)
        # r = max(list(state['start_lane_vehicle_count'].values()))
        r = max(list(state['start_lane_waiting_vehicle_count'].values()))
        return -r

    # def get_score(self):
    #     score = {id_: self._get_score(id_) for id_ in self.intersection_id}
    #     score = self.dict_Avg(score)
    #     return score

    # def _get_score(self, id_):
    #     state = self.intersection_info(id_)
    #     start_lane_speed = state['start_lane_speed']
    #     end_lane_speed = state['end_lane_speed']
    #     score = (self.dict_Avg(start_lane_speed) + self.dict_Avg(end_lane_speed)) / 2
    #     # score = (1 / (1 + np.exp(-1 * x))) / self.num_step
    #     return score

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def dict_Avg(self, Dict):
        Len = len(Dict)  # 取字典中键值对的个数
        Sum = sum(Dict.values())  # 取字典中键对应值的总和
        Avg = Sum / Len
        return Avg

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = self.roadnetFile
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)

            # build the info dict for intersections
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None, }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys():  # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road + "_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']:  # use geo-distance
                    row = np.array([0] * total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = self._cal_distance(location_1, location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else:  # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']:  ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)

                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

            path_to_save = os.path.join(self.record_dir, 'traffic_light_node_dict.conf')
            with open(path_to_save, 'w') as f:
                f.write(str(traffic_light_node_dict))
                print("\033[1;33mSaved traffic_light_node_dict\033[0m")

        return traffic_light_node_dict

    def _update_leave_entering_approach_vehicle(self, id_):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step[id_]:
            for lane in self.lane_phase_info[id_]['start_lane']:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.lane_phase_info[id_]['start_lane']:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[id_][lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[id_][lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )

        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, id_, list_vehicle_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time[id_]:
                self.dic_vehicle_arrive_leave_time[id_][vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def _update_left_time(self, id_, list_vehicle_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[id_][vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a - b) ** 2))

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, path_to_log):
        for id_ in self.intersection_id:
            # print("log for ", id_)
            path_to_log_file = os.path.join(path_to_log, "vehicle_{0}.csv".format(id_))
            dic_vehicle = self.get_dic_vehicle_arrive_leave_time(id_)
            df = pd.DataFrame.from_dict(dic_vehicle, orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

            # path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            # f = open(path_to_log_file, "wb")
            #
            # # Use pickle to pack data flow into
            # pickle.dump(self.list_inter_log[inter_ind], f)
            # f.close()

    def get_dic_vehicle_arrive_leave_time(self, id_):
        return self.dic_vehicle_arrive_leave_time[id_]
