from Carla_sac import get_args
import argparse
import datetime
import os
import pprint
import numpy as np
import torch
# from mujoco_env import make_mujoco_env
# from tianshou.env import SubprocVectorEnv

from RoundaboutCarlaEnv import RoundaboutCarlaEnv 
# from ac_zhy.wrappers_missile import WraSingleAgent
# from config import Algo_config, Env_config_missile
# from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from utils import AttackedActorProb

from rich.pretty import pprint


def eval_sac(args=get_args()):
    # env, train_envs, test_envs = make_mujoco_env(
    #     args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    # )
    env = RoundaboutCarlaEnv(port=args.port)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape, "Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = AttackedActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
        noise=args.noise
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if not args.resume_path:
        print("resume_path is not provided, use default path.")
        # log_path =  r"/home/yq/CARLA_0.9.6/CarlaRL/gym-carla_lanekeep/log/Carla-KeepLane-v3/sac/0/221024-151433/policy.pth" # 偏右行驶
        # args.resume_path = '/home/zhouxinning/Workspace/Carla/CARLA_RL/gym-carla_lanekeep/log/Baselines/Carla_SAC_KeepLane/checkpoint17.pth' #震荡
        args.resume_path = '/home/zhouxinning/Workspace/Carla/CARLA_RL/gym-carla_lanekeep/log/Baselines/Carla_SAC_KeepLane/checkpoint21.pth'
    print("Loaded agent from: ", args.resume_path)
    policy.load_state_dict(torch.load(args.resume_path, map_location=torch.device(args.device))['model'])

    # Let's watch its performance!
    policy.eval()
    # policy.actor.forward()
    # env.seed(args.seed)
    # "100" should match with max_timesteps in RoundaboutCarlaEnv
    buffer = ReplayBuffer(size=args.test_num * 100)
    collector = Collector(policy, env, buffer)
    # args.render = 0.001
    # result = collector.collect(n_episode=100, render=args.render)
    result = collector.collect(n_episode=args.test_num, render=args.render, no_grad=False)
    
    print(f'Final reward for {args.test_num} trajectory:' )

    pprint({
        'rewards': f'{result["rews"].mean():.2f} / {np.std(result["rews"]):.2f}',
        'length': f'{result["lens"].mean():.2f} / {np.std(result["lens"]):.2f}',
        'velocity': f'{collector.buffer.info.velocity.mean():.2f} / {collector.buffer.info.velocity.std():.2f}',
        'deviation': f'{collector.buffer.info.deviation.mean():.2f} / {collector.buffer.info.deviation.std():.2f}',
        'is_crash': f'{collector.buffer.info.is_crash.mean():.2f} / {collector.buffer.info.is_crash.std():.2f}',
    })


if __name__ == "__main__":
    eval_sac()
