import os
import json
from easydict import EasyDict

import sys
sys.path.append('/Users/changhee/Lab/RL/marlenv_hotfix_copy/marlenv')
sys.path.append('/Users/changhee/Lab/RL/marlenv_hotfix_copy/rl2')

import marlenv
from marlenv.wrappers import make_snake
import torch

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers.multi_agent import MAMaxStepWorker, MAEpisodicWorker

# FIXME: Remove later
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def ppo(obs_shape, ac_shape, config, props, load_dir=None):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=False,
                     discrete=True,
                     reorder=True,#props.reorder,
                     optimizer='torch.optim.RMSprop',
                     high=props.high)
    if load_dir is not None:
        model.load(load_dir)
    agent = PPOAgent(model,
                     train_interval=config.train_interval,
                     #  n_env=props.n_env,
                     n_env=props.num_envs,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    # 'n_env': props.n_env})
                                    'n_env': props.num_envs})
    return agent


def train(config):
    logger = Logger(name='PPOCOOP', args=config, log_dir='/Users/changhee/Lab/RL/marlenv_hotfix_copy/rl2/rl2/examples/runs/DATA/Coop'+'_'+str(config.num_snakes)+'_'+str(config.custom_rewardf.lose)+'_'+str(config.custom_rewardf.kill)+'_'+str(config.custom_rewardf.time))
    env, observation_shape, action_shape, props = make_snake(
        env_id='SnakeCoop-v1',
        num_envs=config.n_env,
        num_snakes=config.num_snakes,
        num_fruits=config.num_snakes * 2,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=config.custom_rewardf,
        # render_mode='rgb_array',
    )
    props = EasyDict(props)

    agents = []
    for _ in range(config.num_snakes):
        agents.append(ppo(observation_shape, action_shape, config, props))

    worker = MAMaxStepWorker(
        env, props.num_envs, agents,                     
        max_steps=config.max_steps,
        # max_steps=int(1e8),
        training=True,
        logger=logger,
        log_interval=config.log_interval,
        render=True,
        # render_mode='rgb_array',
        render_interval=50000,
        is_save=True,
        save_interval=config.save_interval,
    )
    # with torch.autograd.set_detect_anomaly(True):
    worker.run()     

    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if load_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_dir = os.path.join(load_dir, "ckpt")
        with open(config_file, "r") as config_f:
            _config = EasyDict(json.load(config_f))
    else:
        model_dir = None
    logger = Logger(name='PPOCOOP', args=config, log_dir='/Users/changhee/Lab/RL/marlenv_hotfix_copy/rl2/rl2/examples/runs/EP_DATA/Coop'+'_'+str(config.num_snakes)+'_'+str(config.custom_rewardf.kill)+'_'+str(config.custom_rewardf.time))

    env, observation_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=config.custom_rewardf
    )
    props = EasyDict(props)

    agents = []
    for i in range(config.num_snakes):
        # if model_dir is not None:
        #     model_file = os.path.join(model_dir,
        #                               f'agent{i}', '100k', 'PPOModel.pt')
        # else:
        #     model_file = None
        agents.append(
            # ppo(observation_shape, action_shape, config, props,
            #     load_dir=model_file)
            ppo(observation_shape, action_shape, config, props,
                load_dir='/Users/changhee/Lab/RL/marlenv_hotfix_copy/rl2/rl2/examples/runs/DATA/Coop'+'_'+str(config.num_snakes)+'_'+str(config.custom_rewardf.kill)+'_'+str(config.custom_rewardf.time)+'/ckpt/agent'+str(i)+'/1000k/PPOModel.pt')
        )

    worker = MAEpisodicWorker(env, props.num_envs, agents,
                              max_episodes=1, training=False,
                              render=True,
                              render_interval=1,
                              logger=logger,
                              data_dir='/Users/changhee/Lab/RL/marlenv_hotfix_copy/rl2/rl2/examples/runs/EP_DATA/Coop'+'_'+str(config.num_snakes)+'_'+str(config.custom_rewardf.kill)+'_'+str(config.custom_rewardf.time))
    worker.run()


if __name__ == "__main__":
    myconfig = {
        'n_env': 64,
        'num_snakes': 2,
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'batch_size': 512,
        'epoch': 4,
        'max_step': int(1e6),
        'train_interval': 512,
        'log_level': 10,
        'log_interval': 5000,
        'save_interval': 100000,
        'lr': 1e-4,
        'gamma': 0.99,
        'grad_clip': 10,
        'tag': 'DATA',
        'custom_rewardf': {
            'fruit': 1.0,
            'kill': -1.0,
            'lose': -10.0,
            'win': 0.0,
            'time': -0.01
        }
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    # test(config)
    
