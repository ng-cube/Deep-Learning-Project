import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import random, math
from itertools import count
import utils, os

class Model(nn.Module):
    def __init__(self, cfg, n_actions, device, output_dir="tmp"):
        super().__init__()

        # set variables 
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device
        self.output_dir = output_dir

        # instantiate the actual model 
        self.policy_net = DQN(
            cfg=cfg,
            n_actions=n_actions
        ).to(self.device)

        self.target_net = DQN(
            cfg=cfg,
            n_actions=n_actions
        ).to(self.device)

        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # initialize optimizer & loss
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=cfg.LR
        )
        self.loss = torch.nn.SmoothL1Loss()


    def save(self):
        torch.save(self.policy_net.state_dict(), os.path.join(self.output_dir, "policy_net.pt"))
        torch.save(self.target_net.state_dict(), os.path.join(self.output_dir, "target_net.pt"))

    def get_action(self, state, train=True):
        if train:
            eps_th = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END)* \
                math.exp(-1. * self.steps_done / self.cfg.EPS_DECAY)
            self.steps_done += 1
            if random.random() >= eps_th:
                with torch.no_grad():
                    return self.policy_net(state.to(self.device)).max(1)[1].view(1,1), eps_th
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long), eps_th

        else:
            with torch.no_grad():
                return self.policy_net(state.to(self.device)).max(1)[1].view(1,1)
            

    def train(self, env, memory):
        # initialize tracker df 
        tracker_df = pd.DataFrame(
            columns=[
                'steps_done', 'steps', 'reward', 
                'cum_reward', 'episode_length', 'episode_time'
            ]
        )

        test_df = pd.DataFrame(
            columns=[
                'steps_done', 'reward',
            ]
        )
        # initialize variables
        self.steps_done = 0 
        self.learn_counter = 0

        # Enforce Atari 100k step limit
        while self.steps_done <= self.cfg.TOTAL_STEPS:
            obs, _ = env.reset()
            state = utils.get_state(obs)
            total_reward = 0.0 
            for t in count():
                action, epsilon = self.get_action(state, train=True)

                # execute environment step 
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                next_state = None if done else utils.get_state(obs)
                reward = torch.tensor([reward], device=self.device)

                if self.cfg.BIN_REWARD:
                    reward = torch.sign(reward)  

                # push to memory
                if next_state is not None:
                    memory.push(
                        state,
                        action.to('cpu'),
                        reward.to('cpu'),
                        next_state,
                        done
                    )        

                state = next_state 

                if done:
                    # store results in tracker df & break
                    cum_reward, episode_length, episode_time = info['episode']['r'][0], info['episode']['l'][0], info['episode']['t'][0]
                    tracker_df.loc[len(tracker_df)] = [self.steps_done, t, total_reward, cum_reward, episode_length, episode_time]

                    # also run one evaulation episode
                    test_score = self._test(env)
                    test_df.loc[len(test_df)] = [self.steps_done, test_score]
                    print(f"Test score: {test_score}")
                    break
            
                if not self.steps_done % 1_000:
                    # print most recent row from the tracker df 
                    if len(tracker_df) > 0:
                        steps_done, t, total_reward, cum_reward, episode_length, episode_time = tracker_df.iloc[-1]
                        print(f"Steps done: {steps_done} \t Total reward: {total_reward} \t Cumulative reward: {cum_reward} \t Episode length: {episode_length} \t Episode time: {episode_time} \t Epsilon: {epsilon}")

                # optimize the model every 4 steps for 1 mini-batch, but only if the inital memory is full
                if not self.steps_done % 4 and self.steps_done > self.cfg.INITIAL_MEMORY:
                    # optimize the model 
                    loss = self._optimize_model(memory)
                    if not self.steps_done % 400:
                        print(f"Loss: {loss}")


        # close the environment
        env.close()

        # store the tracker_df 
        tracker_df.to_csv(os.path.join(self.output_dir, "tracker.csv")) 
        test_df.to_csv(os.path.join(self.output_dir, "test.csv"))


    def _test(self, env):
        obs, _ = env.reset()
        state = utils.get_state(obs)
        total_reward = 0.0
        done = False
        while not done:
            action = self.get_action(state, train=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = utils.get_state(obs)
        return total_reward


    def _optimize_model(self, memory, n_iter=1):
        if len(memory) < self.cfg.BATCH_SIZE:
            return
        print(f"Optimizing model for {n_iter} iterations")
        avg_loss = 0
        for i in range(n_iter):
            print(f"Optimization step {i+1}/{n_iter}")
            # sample batch from memory 
            state, action, reward, state_, done = memory.sample(self.cfg.BATCH_SIZE)

            # convert to tensors and concatenate
            state = torch.cat(state).to(self.device)
            action = torch.cat(action).to(self.device)
            state_ = torch.cat(state_).to(self.device)
            reward = torch.cat(reward).to(self.device)
            done = torch.tensor(done, dtype=torch.int).to(self.device)

            # Calcualte the value of the action taken 
            q = self.policy_net(state).gather(1, action.view(-1, 1))
            qmax = self.target_net(state_).max(dim=1)[0]

            nonterminal_target = reward + self.cfg.GAMMA * qmax
            terminal_target = reward
            q_target = (1 - done) * nonterminal_target + done * terminal_target

            loss = self.loss(q.view(-1), q_target)
            avg_loss += loss.item()

            # Perform backward propagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # increment the learn counter
            self.learn_counter += 1 

            # check if target_net should be replaced
            if not self.learn_counter % self.cfg.TARGET_UPDATE:
                print('updating target net')
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # return the average loss for debugging
        return avg_loss / n_iter




class DQN(nn.Module):
    def __init__(self, cfg, n_actions):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cfg.N_STACK, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(6400, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255
        # pass through conv block
        x = self.conv_block(x)
        # flatten
        x = x.view(x.size(0), -1)
        # pass through fully connected layers
        x = self.fc_block(x)
        return x