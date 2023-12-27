import hydra, os
from omegaconf import DictConfig 
from itertools import count
import time 
import numpy as np 
import pandas as pd

import torch 
import torch.nn as nn


import utils 
import gymnasium 

from models import DQN



@hydra.main(config_path="config", config_name="dqn_baseline")
def main(cfg: DictConfig):
    working_dir = os.getcwd()

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # set up environment
    env = gymnasium.make(cfg.ENV_NAME)


    # wrap the environment 
    env = gymnasium.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
    env = gymnasium.wrappers.FrameStack(env, cfg.N_STACK)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)

    # load the actual network (training happens within the network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN.Model(
        cfg=cfg,
        n_actions=env.action_space.n,
        device=device,
        output_dir=output_dir,
    ).to(device)

    # initialize the memory and tracker variables
    memory = utils.ReplayMemory(cfg.INITIAL_MEMORY * cfg.MEMORY_SIZE_FACTOR)

    # train the model 
    tracker_df = model.train(
        env=env,
        memory=memory,
    )
 
    # test the model 
    print('Testing the model')
    test_performance = []
    for episode in range(cfg.test.N_EPISODES):
        obs, _ = env.reset()
        state = utils.get_state(obs)
        total_reward = 0.0 
        for t in count():
            action = model.get_action(
                state.to(device),
                train=False
            )

            if cfg.RENDER_TEST:
                env.render()
                time.sleep(0.02)

            obs, reward, trunc, term, info = env.step(action)
            done = term or trunc
            total_reward += reward 

            next_state = None if done else utils.get_state(obs)
            state = next_state 
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                test_performance.append(total_reward)
                break
    
    # close the environment
    env.close()

    # save model
    model.save()

    # save test performance list as .npy file
    np.save(os.path.join(output_dir, "test_performance.npy"), np.array(test_performance))



if __name__ == "__main__":
    main()