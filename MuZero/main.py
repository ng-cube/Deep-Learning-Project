import hydra, os
from hydra.core.hydra_config import HydraConfig

from absl import app
from omegaconf import DictConfig 
from itertools import count
import time 
import numpy as np 
import pandas as pd

import torch 
import torch.nn as nn
import multiprocessing
import threading

import utils 
import gymnasium 

from utils import create_atari_environment
from MuZero.config import make_atari_config
from MuZero.models import MuZeroAtariNet
from MuZero.replay import PrioritizedReplay
from MuZero.pipeline import (
    run_data_collector,
    run_training,
    run_evaluator,
    run_self_play,
)


@hydra.main(config_path="config", config_name="muzero")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    cfg.output_dir = output_dir


    # get device & set random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.operational.seed)
    random_state = np.random.RandomState(cfg.operational.seed) 

    
    # build the atari environment 
    def environment_builder():
        return create_atari_environment(
            env_name=cfg.environment.name,
            seed=random_state.randint(1, 2**32),
            frame_skip=cfg.environment.frame_skip,
            frame_stack=cfg.environment.frame_stack,
            screen_height=cfg.environment.height,
            screen_width=cfg.environment.width,
            noop_max=cfg.environment.noop_max,
            max_episode_steps=cfg.environment.max_episode_steps,
            terminal_on_life_loss=cfg.environment.terminal_on_life_loss,
            clip_reward=cfg.environment.clip_reward,
            episodic_life=cfg.environment.episodic_life,
        )


    # stack the environments
    self_play_envs = [environment_builder() for _ in range(cfg.operational.num_actors)]
    eval_env = environment_builder()


    # extract relevant information from env
    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n
    tag = self_play_envs[0].spec.id
    
    # create the config 
    config = make_atari_config(
        num_training_steps=cfg.operational.num_training_steps,
        batch_size=cfg.operational.batch_size,
        min_replay_size=cfg.memory.min_replay_size,
        use_tensorboard=cfg.operational.use_tensorboard,
        clip_grad=cfg.operational.clip_grads,
    )


    # load the networks
    network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size
    )

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.lr_init,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_decay_rate,
    )

    actor_network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size
    )
    actor_network.share_memory()
    new_ckpt_network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size
    )


    # initialize replay memory
    replay = PrioritizedReplay(
        cfg.memory.replay_capacity,
        cfg.memory.priority_exponent,
        cfg.memory.importance_sampling_exponent,
        random_state,
    )


    # set up multiprocessing
    stop_event = multiprocessing.Event()
    data_queue = multiprocessing.SimpleQueue()
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Shared training Steps counter, so actors can adjust temperature used in MCTS.
    train_steps_counter = multiprocessing.Value('i', 0)

    # Start to collect samples from self-play on a new thread.
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay, cfg.operational.samples_save_frequency, cfg.operational.samples_save_dir, tag),
    )
    data_collector.start()


    # start the main training loop on a new thread
    learner = threading.Thread(
        target=run_training,
        args=(
            config,
            network,
            optimizer,
            lr_scheduler,
            device,
            actor_network,
            replay,
            data_queue,
            train_steps_counter,
            os.path.join(cfg.output_dir, "checkpoint_dir"),
            checkpoint_files,
            stop_event,
            tag,
        ),
    )
    learner.start()


    # start evaluation loop on a separate process.
    evaluator = multiprocessing.Process(
        target=run_evaluator,
        args=(
            config,
            new_ckpt_network,
            device,
            eval_env,
            0.0,
            checkpoint_files,
            stop_event,
            tag,
        ),
    )
    evaluator.start()


    ## Start self-play process
    actors = []
    for i in range(cfg.operational.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                config,
                i,
                actor_network,
                device,
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                stop_event,
                tag,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    learner.join()
    data_collector.join()
    evaluator.join()


if __name__ == '__main__':
    
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    main()
    #app.run(main)

