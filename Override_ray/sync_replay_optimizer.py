from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections
import numpy as np
import math
import pandas as pd

import ray
from Override_ray.replay_buffer import ReplayBuffer, \
    PrioritizedReplayBuffer
from Override_ray.policy_optimizer import PolicyOptimizer
from Override_ray.metrics import get_learner_stats
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.compression import pack_if_needed
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.schedules import LinearSchedule
from ray.rllib.utils.memory import ray_get_and_free

from torch.optim import Adam

from utils.misc import *
from utils.utility import policy_id_handle

from gym_cityflow.envs.cityflow_env_ray import CityFlowEnvRay as env

logger = logging.getLogger(__name__)

j_store = 0


class SyncReplayOptimizer(PolicyOptimizer):
    """Variant of the local sync optimizer that supports replay (for DQN).

    This optimizer requires that rollout workers return an additional
    "td_error" array in the info return of compute_gradients(). This error
    term will be used for sample prioritization."""

    def __init__(self,
                 workers,
                 config,
                 learning_starts=1000,
                 buffer_size=50000,
                 prioritized_replay=True,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 schedule_max_timesteps=100000,
                 beta_annealing_fraction=0.2,
                 final_prioritized_replay_beta=0.4,
                 train_batch_size=32,
                 sample_batch_size=4,
                 before_learn_on_batch=None,
                 synchronize_sampling=False):
        """Initialize an sync replay optimizer.

        Arguments:
            workers (WorkerSet): all workers
            learning_starts (int): wait until this many steps have been sampled
                before starting optimization.
            buffer_size (int): max size of the replay buffer
            prioritized_replay (bool): whether to enable prioritized replay
            prioritized_replay_alpha (float): replay alpha hyperparameter
            prioritized_replay_beta (float): replay beta hyperparameter
            prioritized_replay_eps (float): replay eps hyperparameter
            schedule_max_timesteps (int): number of timesteps in the schedule
            beta_annealing_fraction (float): fraction of schedule to anneal
                beta over
            final_prioritized_replay_beta (float): final value of beta
            train_batch_size (int): size of batches to learn on
            sample_batch_size (int): size of batches to sample from workers
            before_learn_on_batch (function): callback to run before passing
                the sampled batch to learn on
            synchronize_sampling (bool): whether to sample the experiences for
                all policies with the same indices (used in MADDPG).
        """
        PolicyOptimizer.__init__(self, workers)

        self.replay_starts = learning_starts
        # linearly annealing beta used in Rainbow paper
        self.prioritized_replay_beta = LinearSchedule(
            schedule_timesteps=int(
                schedule_max_timesteps * beta_annealing_fraction),
            initial_p=prioritized_replay_beta,
            final_p=final_prioritized_replay_beta)
        self.prioritized_replay_eps = prioritized_replay_eps
        self.train_batch_size = train_batch_size
        self.before_learn_on_batch = before_learn_on_batch
        self.synchronize_sampling = synchronize_sampling

        # Stats
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.learner_stats = {}

        '''Attention Info'''
        self.traffic_light_node_dict = {}
        self.record_dir = '/home/skylark/PycharmRemote/Gamma-Reward-Perfect/record/' + config["env_config"]["Name"]
        self.read_traffic_light_node_dict()
        self.tmp_dic = self.traffic_light_node_dict['intersection_1_1']['inter_id_to_index']
        # -------------------------------------------

        '''
        For compare reward change 
        '''
        self.raw_reward_store = {}
        self.Reward_store = {}
        for inter_id in self.tmp_dic:
            self.raw_reward_store[inter_id] = []
            self.Reward_store[inter_id] = []
        # self.j_store = 0
        # ------------------------------
        # Set up replay buffer
        if prioritized_replay:
            def new_buffer():
                return PrioritizedReplayBuffer(
                    buffer_size, alpha=prioritized_replay_alpha)
        else:
            def new_buffer():
                return ReplayBuffer(buffer_size)

        self.replay_buffers = collections.defaultdict(new_buffer)

        if buffer_size < self.replay_starts:
            logger.warning("buffer_size={} < replay_starts={}".format(
                buffer_size, self.replay_starts))

        '''
        For Gamma Reward by Skylark
        '''
        self.memory_thres = config["env_config"]["memory_thres"]
        self.num_steps_presampled = 0
        self.gamma = 0.5
        self.index = 0
        self.punish_coeff = 1.5
        self.config = config
        # Set up replay buffer
        if prioritized_replay:

            def pre_new_buffer():
                return PrioritizedReplayBuffer(
                    buffer_size + self.memory_thres, alpha=prioritized_replay_alpha)
        else:

            def pre_new_buffer():
                return ReplayBuffer(buffer_size + self.memory_thres)
        self.pre_replay_buffers = collections.defaultdict(pre_new_buffer)
        # ------------------------------------------

        # '''
        # For Attention Reward by Skylark
        # '''
        # sa_size = [(15, 8), (15, 8), (15, 8), (15, 8), (15, 8), (15, 8)]
        # critic_hidden_dim = 128
        # attend_heads = 4
        # q_lr = 0.01
        # self.attention = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
        #                                  attend_heads=attend_heads)
        # self.target_attention = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
        #                                         attend_heads=attend_heads)
        # hard_update(self.target_attention, self.attention)
        # self.attention_optimizer = Adam(self.attention.parameters(), lr=q_lr,
        #                                 weight_decay=1e-3)
        # self.niter = 0
        # ------------------------------------------------------------------

    @override(PolicyOptimizer)
    def step(self, attention_score_dic=None):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                batch = SampleBatch.concat_samples(
                    ray_get_and_free([
                        e.sample.remote()
                        for e in self.workers.remote_workers()
                    ]))
            else:
                batch = self.workers.local_worker().sample()

            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)
            '''
            For Gamma Reward by LJJ (You can check the local history for changing)
            '''
            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    self.pre_replay_buffers[policy_id].add(
                        pack_if_needed(row["obs"]),
                        row["actions"],
                        row["rewards"],
                        pack_if_needed(row["new_obs"]),
                        row["dones"],
                        weight=None)

            if self.num_steps_presampled >= self.memory_thres:
                self._preprocess(batch, attention_score_dic)

            self.num_steps_presampled += batch.count

        # -----------------------------------------------------------------------

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput,
                                             3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })

    def _optimize(self):
        samples = self._replay()

        with self.grad_timer:
            if self.before_learn_on_batch:
                samples = self.before_learn_on_batch(
                    samples,
                    self.workers.local_worker().policy_map,
                    self.train_batch_size)
            info_dict = self.workers.local_worker().learn_on_batch(samples)
            for policy_id, info in info_dict.items():
                self.learner_stats[policy_id] = get_learner_stats(info)
                replay_buffer = self.replay_buffers[policy_id]
                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    td_error = info["td_error"]
                    new_priorities = (
                            np.abs(td_error) + self.prioritized_replay_eps)
                    replay_buffer.update_priorities(
                        samples.policy_batches[policy_id]["batch_indexes"],
                        new_priorities)
            self.grad_timer.push_units_processed(samples.count)

        self.num_steps_trained += samples.count

    def _replay(self):
        samples = {}
        idxes = None
        with self.replay_timer:
            for policy_id, replay_buffer in self.replay_buffers.items():
                if self.synchronize_sampling:
                    if idxes is None:
                        idxes = replay_buffer.sample_idxes(
                            self.train_batch_size)
                else:
                    idxes = replay_buffer.sample_idxes(self.train_batch_size)

                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    (obses_t, actions, rewards, obses_tp1, dones, weights,
                     batch_indexes) = replay_buffer.sample_with_idxes(
                        idxes,
                        beta=self.prioritized_replay_beta.value(
                            self.num_steps_trained))
                else:
                    (obses_t, actions, rewards, obses_tp1,
                     dones) = replay_buffer.sample_with_idxes(idxes)
                    weights = np.ones_like(rewards)
                    batch_indexes = -np.ones_like(rewards)
                samples[policy_id] = SampleBatch({
                    "obs": obses_t,
                    "actions": actions,
                    "rewards": rewards,
                    "new_obs": obses_tp1,
                    "dones": dones,
                    "weights": weights,
                    "batch_indexes": batch_indexes
                })
        return MultiAgentBatch(samples, self.train_batch_size)

    def _preprocess(self, batch, attention_score_dic=None):
        """
        Self-defined function: For Gamma Reward Replay Buffer Amendment
        :param batch: SampleBatch class,
        :param attention_score_dic: For transferring Attention score calculated by target attention layers
        :return: return Amendatory Replay Buffer
        """
        global j_store
        for policy_id, s in batch.policy_batches.items():
            storage = list(self.pre_replay_buffers[policy_id]._storage)
            index = len(storage) - self.memory_thres - 1
            tmp_buffer = storage.copy()
            current_intersection = self.inter_num_2_id(policy_id_handle(policy_id))
            '''
            For comparing the change of rewards 
            '''
            # ------------------------------
            while index > self.index - 1:
                obs = storage[index][0]
                action = storage[index][1]
                reward = storage[index][2]
                new_obs = storage[index][3]
                done = storage[index][4]
                p_value = 0

                all_roads_path_2dlst = np.array(
                    self.config['env_config']['lane_phase_info'][current_intersection][
                        'phase_roadLink_mapping'][action + 1])
                all_end_roads = self.config['env_config']['lane_phase_info'][current_intersection]['end_lane']
                permitted_end_roads = np.unique([all_roads_path_2dlst[lane_index, 1] for lane_index, start_lane in
                                                 enumerate(all_roads_path_2dlst[:, 0]) if start_lane[-1] != '2'])
                dis_permitted_end_roads = list(set(all_end_roads).difference(set(list(permitted_end_roads))))

                # Take neighbors into account
                for other_policy_id, s in batch.policy_batches.items():
                    other_intersection = self.inter_num_2_id(policy_id_handle(other_policy_id))
                    if other_policy_id != policy_id and other_intersection in \
                            self.traffic_light_node_dict[current_intersection]['neighbor_ENWS']:
                        other_storage = self.pre_replay_buffers[other_policy_id]._storage
                        '''
                        For corresponding lane in a neighbouring intersection, m_2 represents the waiting count in
                        t+n time step and m_1 for t step.  m_2-m_1/m_1
                        '''
                        road_index_dict = {road: road_index for road_index, road in
                                           enumerate(self.config['env_config']['road_sort'][other_intersection])}

                        # differential = np.max(
                        #     np.array(other_storage[index + self.memory_thres - 1:index + self.memory_thres])[:,
                        #     2]) / other_storage[index][2]
                        for road in road_index_dict.keys():
                            if road in all_end_roads:
                                if road in permitted_end_roads:
                                    I_a = -1
                                elif road in dis_permitted_end_roads:
                                    I_a = 0
                                else:
                                    print('wrong')
                                road_index = road_index_dict[road]

                                m_1 = np.array(other_storage[index])[0][road_index]
                                m_2 = np.mean(
                                    [other_storage[index + self.memory_thres - 2][0][road_index],
                                     other_storage[index + self.memory_thres - 1][0][road_index]])
                                if m_2 - m_1 == 0 or m_1 == 0:
                                    differential = 0
                                else:
                                    differential = m_2 - m_1 / m_1  # m_2 = 0, m_1 != 0 -> differential = -1
                                    if differential > 1:
                                        differential = 0

                                p_value += m_1 * np.tanh(differential) * I_a

                if self.config['env_config']['Gamma_Reward']:
                    p_reward = reward + self.gamma * p_value
                    # print('Reward: ' + str(Reward) + ',' + 'reward: ' + str(reward))
                    if p_reward <= -20:
                        p_reward = -20
                    # print(Reward)
                else:
                    p_reward = reward
                '''
                For compare reward change 
                '''
                # if 50 < j_store < 100:
                #     self.raw_reward_store[self.inter_num_2_id(policy_id_handle(policy_id))].append(reward)
                #     self.Reward_store[self.inter_num_2_id(policy_id_handle(policy_id))].append(Reward)

                # ------------------------------
                tmp_buffer[index] = list(storage[index])
                tmp_buffer[index][2] = p_reward
                index -= 1

            for i in range(self.index, len(tmp_buffer) - self.memory_thres):
                self.replay_buffers[policy_id].add(
                    obs_t=tmp_buffer[i][0],
                    action=tmp_buffer[i][1],
                    reward=tmp_buffer[i][2],
                    obs_tp1=tmp_buffer[i][3],
                    done=tmp_buffer[i][4],
                    weight=None)

        # Reward MDP
        index = len(storage) - self.memory_thres - 1
        while index > self.index - 1:
            for policy_id, s in batch.policy_batches.items():
                current_intersection = self.inter_num_2_id(policy_id_handle(policy_id))
                storage = list(self.replay_buffers[policy_id]._storage)
                p_reward = storage[index][2]
                sum_other_reward = 0
                for other_policy_id, s in batch.policy_batches.items():
                    other_intersection = self.inter_num_2_id(policy_id_handle(other_policy_id))
                    if other_policy_id != policy_id and other_intersection in \
                            self.traffic_light_node_dict[current_intersection]['neighbor_ENWS']:
                        other_storage = self.replay_buffers[other_policy_id]._storage
                        pre_other_storage = self.pre_replay_buffers[other_policy_id]._storage
                        if index + self.memory_thres >= len(other_storage):
                            sum_other_reward = 0
                        else:
                            sum_other_reward += np.tanh(other_storage[index + self.memory_thres][2] /
                                                        pre_other_storage[index + self.memory_thres][2]
                                                        - self.punish_coeff)
                Reward = p_reward + self.gamma * sum_other_reward
                self.replay_buffers[policy_id]._storage[index] = list(self.replay_buffers[policy_id]._storage[index])
                self.replay_buffers[policy_id]._storage[index][2] = Reward
                self.replay_buffers[policy_id]._storage[index] = tuple(self.replay_buffers[policy_id]._storage[index])
            index -= 1

        j_store += 1
        self.index = len(storage) - self.memory_thres

        # if j_store == 100:
        #     print("Start recording the reward !!!!!!!!!!!")
        #     raw_reward_store_np = {}
        #     Reward_store_np = {}
        #     for inter_id in self.tmp_dic:
        #         raw_reward_store_np[inter_id] = np.array(self.raw_reward_store[inter_id])
        #         Reward_store_np[inter_id] = np.array(self.Reward_store[inter_id])
        #     raw_reward_store_pd = pd.DataFrame(dict((k, pd.Series(v)) for k, v in raw_reward_store_np.items()))
        #     Reward_store_pd = pd.DataFrame(dict((k, pd.Series(v)) for k, v in Reward_store_np.items()))
        #     raw_reward_store_pd.to_csv(os.path.join(self.record_dir, 'raw_reward_store_pd.csv'))
        #     Reward_store_pd.to_csv(os.path.join(self.record_dir, 'Reward_store_pd.csv'))

        # self.replay_buffers = storage[:self.index]

        self.num_steps_sampled = len(self.replay_buffers[policy_id]._storage)  # Any policy_id is OK

        if self.num_steps_sampled >= self.replay_starts:
            self._optimize()

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def read_traffic_light_node_dict(self):
        path_to_read = os.path.join(self.record_dir, 'traffic_light_node_dict.conf')
        with open(path_to_read, 'r') as f:
            self.traffic_light_node_dict = eval(f.read())
            print("Read traffic_light_node_dict")

    def inter_num_2_id(self, num):
        return list(self.tmp_dic.keys())[list(self.tmp_dic.values()).index(num)]

    # def attention_update(self, sample, logger=None):
    #     """
    #     For Attention
    #     """
    #     obs, acs, rews, next_obs, dones = sample
    #     attention_in = list(zip(obs, acs))
    #     attention_rets = self.attention(attention_in, regularize=True,
    #                                     logger=logger, niter=self.niter)
    #     
    #     return attention_rets
