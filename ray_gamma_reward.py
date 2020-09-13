import argparse
import json
import logging
import os
from datetime import datetime

import gym
import ray
import ray.rllib.agents.qmix as qmix
from gym.spaces import Tuple
from ray import tune
from ray.rllib.agents.qmix import QMixTrainer
# from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.logger import pretty_print

# from model_gamma_reward import MyModelClass
import Override_ray.dqn as dqn
from Model.custom_keras_model import MyKerasQModel
from Override_ray.dqn import DQNTrainer
from Override_ray.dqn_policy import DQNTFPolicy
from gym_cityflow.envs.cityflow_env_ray import CityFlowEnvRay
from utils.utility import parse_roadnet


# ray.utils.set_cuda_visible_devices("1")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gym.logger.set_level(40)


def env_config(args):
    config = {}
    config["threshold"] = args.threshold
    config["Name"] = args.project_name
    config["Gamma_Reward"] = args.Gamma_Reward
    config["MIN_ACTION_TIME"] = args.MIN_ACTION_TIME
    config["memory_thres"] = args.memory_thres

    global_config = json.load(open('/home/skylark/PycharmRemote/Gamma-Reward-Perfect/config/global_config.json'))
    cityflow_config = json.load(open(global_config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    config['road_sort'] = {}
    for id_ in list(config['lane_phase_info'].keys()):
        config['road_sort'][id_] = config["lane_phase_info"][id_]['start_lane']

    print('\033[1;35mThe project name is {}\033[0m'.format(config["Name"]))
    return config


def agent_config(args, num_agents, obs_space, act_space, config_env, num_row):
    ray.utils.set_cuda_visible_devices(args.gpu_id)
    if args.tune:
        config = {}
    elif args.mod == 'DQN':
        config = dqn.DEFAULT_CONFIG.copy()
        config["env"] = CityFlowEnvRay
        config["noisy"] = True
        config["dueling"] = args.dueling
        config["double_q"] = args.double_q
        config["num_cpus_per_worker"] = 1
        config["num_gpus"] = args.num_gpus
        config["num_workers"] = args.num_workers
        config["multiagent"] = {
            "policies": {
                "policy_{}".format(i): (DQNTFPolicy, obs_space, act_space, {"agent_id": i, })
                for i in range(num_agents)
            },
            "policy_mapping_fn":
                tune.function(lambda agent_id:
                              "policy_{}".format((int(agent_id.split('_')[1]) - int(1)) *
                                                 int(num_row) + int(agent_id.split('_')[2]) - int(1)))
        }

        config["model"] = {
            "custom_model": "MyKerasQModel",
            "custom_options": {"use_lstm": True},
        }
    elif args.mod == 'QMIX':
        config = qmix.DEFAULT_CONFIG.copy()
        config["env"] = "cityflow_multi"

    config["env_config"] = config_env
    config["sample_batch_size"] = 72
    config["timesteps_per_iteration"] = 360
    config["target_network_update_freq"] = 5
    config["horizon"] = 360
    config["train_batch_size"] = 100
    config["exploration_fraction"] = 0.8
    config["exploration_final_eps"] = 0.1
    config["no_done_at_end"] = True
    config["learning_starts"] = 100
    config["eager_tracing"] = True
    config["compress_observations"] = False
    config["lr"] = args.lr
    # config["clip_rewards"] = True
    # print("compress_observations" + str(config["compress_observations"]))

    return config


def main(args):
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_env = env_config(args)

    global_config = json.load(open('/home/skylark/PycharmRemote/Gamma-Reward-Perfect/config/global_config.json'))
    roadnet = global_config['cityflow_config_file']
    roadnet_list = roadnet.split('_')
    num_row = int(roadnet_list[1])
    num_col = int(roadnet_list[2].split('.')[0])
    print('\033[1;35mThe scale of the current roadnet is {} x {}\033[0m'.format(num_row, num_col))

    num_agents = num_row * num_col

    if args.mod == 'DQN':
        obs_space = CityFlowEnvRay.observation_space
        act_space = CityFlowEnvRay.action_space
        ModelCatalog.register_custom_model("MyKerasQModel", MyKerasQModel)
    elif args.mod == 'QMIX':
        grouping = {
            "group_1": [id_ for id_ in CityFlowEnvRay.intersection_id]
        }
        obs_space = Tuple([
            CityFlowEnvRay.observation_space for _ in range(num_agents)
        ])
        act_space = Tuple([
            CityFlowEnvRay.action_space for _ in range(num_agents)
        ])
        register_env(
            "cityflow_multi",
            lambda config_: CityFlowEnvRay(config_).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))
    config_agent = agent_config(args, num_agents, obs_space, act_space, config_env, num_row)

    ray.init(local_mode=False, redis_max_memory=1024 * 1024 * 40, temp_dir='/home/skylark/log/')
    if args.tune:  # False
        tune.run(
            "DQN",
            stop={
                "timesteps_total": 400000,
            },
            config=config_agent,
            checkpoint_freq=2,
            checkpoint_at_end=True,
        )
    else:
        if args.mod == 'DQN':  # True
            trainer = DQNTrainer(config=config_agent,
                                 env=CityFlowEnvRay)

            for i in range(args.n_epoch):
                # Perform one iteration of training the policy with DQN
                result = trainer.train()
                print(pretty_print(result))
        elif args.mod == 'QMIX':  # False
            trainer = QMixTrainer(config=config_agent,
                                  env="cityflow_multi")

            for i in range(args.n_epoch):
                # Perform one iteration of training the policy with DQN
                result = trainer.train()
                print(pretty_print(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='1104_GR_no_end_1_6', help='Name of Project')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--tune', type=bool, default=False, help='Tune or Trainer')
    parser.add_argument('--Gamma_Reward', dest='Gamma_Reward', action='store_true')
    parser.add_argument('--no-Gamma_Reward', dest='Gamma_Reward', action='store_false')  # not the real no_Gamma_reward
    parser.set_defaults(Gamma_Reward=True)  
    parser.add_argument('--num_workers', type=int, default=0, help='Number of rollout workers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpu')
    parser.add_argument('--gpu_id', type=str, default='1', help='GPU ID')
    parser.add_argument('--threshold', type=int, default=20, help='Number of threshold')
    parser.add_argument('--ray_object_store_memory', type=int, default=1024 * 1024 * 15, help='Number of threshold')
    parser.add_argument('--mod', type=str, default='DQN', help='DQN, QMIX, IQL')
    parser.add_argument('--MIN_ACTION_TIME', type=int, default=10, help='interval of each action')
    parser.add_argument('--memory_thres', type=int, default=4, help='decay span')
    parser.add_argument('--dueling', dest='dueling', action='store_true')
    parser.add_argument('--no-dueling', dest='dueling', action='store_false')
    parser.set_defaults(dueling=True)
    parser.add_argument('--double_q', dest='double_q', action='store_true')
    parser.add_argument('--no-double_q', dest='double_q', action='store_false')
    parser.set_defaults(double_q=True)

    args = parser.parse_args()
    print('\033[1;33m{}\033[0m'.format(args))
    main(args)
