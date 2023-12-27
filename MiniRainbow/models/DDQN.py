import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import random, math
from itertools import count
import utils, os


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
            self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)






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
        torch.save(self.policy_net.state_dict(), os.path.join(self.output_dir, "model.pt"))

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
                'steps_done', 'reward'
            ]
        )
        # initialize variables
        self.steps_done = 0 
        self.learn_counter = 0
        episode_steps = 0

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
                episode_steps += 1

                next_state = None if done else utils.get_state(obs)
                reward = torch.tensor([reward], device=self.device)

                if self.cfg.BIN_REWARD:
                    # clip reward to [-1, 1]
                    reward = torch.clamp(reward, -1, 1)
                    #reward = torch.sign(reward)  

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
                    episode_steps = 0

                    # every 10 episodes run a test episode
                    if not len(tracker_df) % 10:
                        test_score = self._test(env)
                        print(f"Test score: {test_score}")
                        test_df.loc[len(test_df)] = [self.steps_done, test_score]
                    break
            
                if not self.steps_done % 1_000 and len(tracker_df) > 0:
                    # print most recent row from the tracker df 
                    if len(tracker_df) > 0:
                        steps_done, t, total_reward, cum_reward, episode_length, episode_time = tracker_df.iloc[-1]
                        print(f"Steps done: {steps_done} \t Total reward: {total_reward} \t Cumulative reward: {cum_reward} \t Episode length: {episode_length} \t Episode time: {episode_time} \t Epsilon: {epsilon}")

                if not self.steps_done % 4 and self.steps_done > self.cfg.INITIAL_MEMORY:
                    # optimize the model 
                    self.policy_net.train()
                    loss = self._optimize_model(memory, 1)
                    if not self.steps_done % 1_000:
                        print(f"Steps done: {self.steps_done} \t Loss: {loss}")
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
        avg_loss = 0
        for i in range(n_iter):
            #print(f"Optimization step {i+1}/{n_iter}")
            # sample batch from memory 
            state, action, reward, state_, done = memory.sample(self.cfg.BATCH_SIZE)

            # convert to tensors and concatenate
            state = torch.cat(state).to(self.device)
            action = torch.cat(action).to(self.device)
            state_ = torch.cat(state_).to(self.device)
            reward = torch.cat(reward).to(self.device)
            done = torch.tensor(done, dtype=torch.int).to(self.device)

            q = self.policy_net(state).gather(1, action.view(-1, 1))   
            qmax = self.target_net(state_).max(dim=1)[0].detach()
            #q_eval = self.policy_net(state).gather(1, action)

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
                print('Updating target network')
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return avg_loss / n_iter


class DQN(nn.Module):
    def __init__(self, cfg, n_actions):
        super().__init__()
        # Modify the CNN block to add Batch Normalization
        self.conv_block = nn.Sequential(
            nn.Conv2d(cfg.N_STACK, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32), # Added Batch Normalization
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), # Added Batch Normalization
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Added another Conv layer
            nn.BatchNorm2d(128), # Added Batch Normalization
            nn.LeakyReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(12800, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, 1)  # This outputs the state value
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(12800, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, n_actions)  # This outputs the advantage
        )

    def forward(self, x):
        x = x.float() / 255
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        return value + advantages - advantages.mean()
    


class DQN_lstm(nn.Module):
    def __init__(self, cfg, n_actions):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(cfg.N_STACK, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
        )

        self.conv_output_size = self._get_conv_output_size(cfg)

        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=cfg.head.HIDDEN_DIM,
            batch_first=True
        )

        self.value_stream = nn.Sequential(
            nn.Linear(cfg.head.HIDDEN_DIM, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.head.HIDDEN_DIM, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255
        x = self.conv_block(x)

        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        return value + advantages - advantages.mean()

    def _get_conv_output_size(self, cfg):
        with torch.no_grad():  
            dummy_input = torch.zeros(1, cfg['N_STACK'], 84, 84)
            dummy_output = self.conv_block(dummy_input)
            return int(torch.numel(dummy_output) / dummy_output.size(0))  # Divide by batch size
        

class Attention(nn.Module):
    def __init__(self, feature_dim, intermediate_dim):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, intermediate_dim)
        self.output_fc = nn.Linear(intermediate_dim, feature_dim)

    def forward(self, x):
        attention = F.relu(self.attention_fc(x))
        attention = torch.sigmoid(self.output_fc(attention))
        return attention * x  # element-wise multiplication


class DQN_attention(nn.Module):
    def __init__(self, cfg, n_actions):
        super().__init__()

        self.attention = Attention(12800, 256) 

        # Modify the CNN block to add Batch Normalization
        self.conv_block = nn.Sequential(
            nn.Conv2d(cfg.N_STACK, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32), # Added Batch Normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), # Added Batch Normalization
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Added another Conv layer
            nn.BatchNorm2d(128), # Added Batch Normalization
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(12800, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, 1)  # This outputs the state value
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(12800, cfg.head.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.head.HIDDEN_DIM, n_actions)  # This outputs the advantage
        )

    def forward(self, x):
        x = x.float() / 255
        features = self.conv_block(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention
        attended_features = self.attention(features)
        
        value = self.value_stream(attended_features)
        advantages = self.advantage_stream(attended_features)
        
        return value + advantages - advantages.mean()