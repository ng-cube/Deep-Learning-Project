import os, json
import time 
import gymnasium
import numpy as np 
import cv2
from PIL import Image

import torch 
import torch.nn as nn 

from transformers import AutoTokenizer, FuyuProcessor, FuyuImageProcessor, FuyuForCausalLM, BitsAndBytesConfig


action_dict = {
    "Nothing": 0,
    "Fire": 1,
    "Up": 2,
    "Right": 3,
    "Left": 4,
    "Down": 5,
}

action_dict_inv = {
    "0": "Nothing",
    "1": "Fire",
    "2": "Up",
    "3": "Right",
    "4": "Left",
    "5": "Down",
}



# create the Fuyu based agent
class FuyuAgent(nn.Module):
    """
    Using the recently released Fuyu model (https://huggingface.co/adept/fuyu-8b)
    """
    def __init__(self, device, manual, model_name="ybelkada/fuyu-8b-sharded"):
        super().__init__()
        self.device = device
        self.dtype = torch.float16

        # load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.processor = FuyuProcessor(
            image_processor=FuyuImageProcessor(),
            tokenizer=self.tokenizer,
        )
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)


        # load the quanitzed model 
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype
            )
        self.model = FuyuForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

        # turn off grads
        for param in self.model.parameters():
            param.requires_grad = False

        # restrict the output space to the action space
        print(f"Token ID for 'Nothing': {self.tokenizer('Nothing', return_tensors='pt')['input_ids']}") # 78631
        print(f"Token ID for 'Fire': {self.tokenizer('Fire', return_tensors='pt')['input_ids']}") # 76793
        print(f"Token ID for 'Up': {self.tokenizer('Up', return_tensors='pt')['input_ids']}") # 77936
        print(f"Token ID for 'Right': {self.tokenizer('Right', return_tensors='pt')['input_ids']}") # 79823
        print(f"Token ID for 'Left': {self.tokenizer('Left', return_tensors='pt')['input_ids']}") # 71374, 73607
        print(f"Token ID for 'Down': {self.tokenizer('Down', return_tensors='pt')['input_ids']}") # 85779
        

        self.action_ids = torch.tensor([78631, 76793, 77936, 79823, 71374, 85779])
        self.text_prompt = manual + "\n\n" 
        self.action_history = ["Action: Fire"] # store the previous actions to provide context to the agent

    def get_action(self, observation):
        """
        Givn the current observation and system prompt, generate the next action.
        Inputs:
            observation (np.array (3, 84, 84, 3)): The stacked current observations (channel is last)
        Outputs:
            action (int): The action to take
        """

        # 1. Process the observation
        img_input = [observation[-1]]

        # 2. Generate the text prompt
        action_history = "Previous actions:"
        for a in self.action_history[-5:]:
            action_history += "\n" + a
        text_prompt = self.text_prompt + action_history + "Please generate a specific strategy of what to do next in the game. Make sure the reason well and explain why you are planning to do what. Be precise, specific and brief. Only suggest one course of action.\tStrategy: " #+ "\nMake sure to not repeat actions unless necessary!" + "\n\nNext Action: "

        # 3. Process the inputs jointly
        model_inputs = self.processor(
            text=text_prompt,
            images=img_input,
            return_tensors="pt"
        )

        # 4. Generate the next action
        # 4.1. Generate a single token
        output = self.model(**model_inputs, return_dict=True)
        logits = output.logits[:, -1, :]

        # 4.2. Limit the output space to the relevant tokens 
        logits = logits[:, self.action_ids]

        # 4.3. Sample an action from the distribution
        action = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        print(f"Action distribution: {torch.softmax(logits, dim=-1)}")

        # 5. Add the action to the action history & return it 
        self.action_history.append(f"Action: {action_dict_inv[str(action[0][0].item())]}")
        return action[0][0]

    

def convert_observation(obs: gymnasium.wrappers.frame_stack.LazyFrames) -> np.array:
    """
    Convert the observation to a torch tensor.
    """
    obs = np.array(obs)
    return obs

if __name__ == "__main__":

    # set parameters
    ENV_NAME = "Centipede"
    create_video = True
    NUM_EPISODES = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the manual
    with open("Centipede.txt", "r") as f:
        manual = f.read()

   
    # initialize the agent 
    agent = FuyuAgent(device=device, manual=manual)

    # initialize the environment & wrap it
    env = gymnasium.make(f"ALE/{ENV_NAME}-v5", render_mode="rgb_array_list") # initialize the environment
    env = gymnasium.wrappers.ResizeObservation(env, (84, 84)) # reshape observations (not necessary for Fuyu)
    env = gymnasium.wrappers.FrameStack(env, 3) # stack the 3 most recent frames (similar to https://arxiv.org/abs/2310.19773) (Not actually necessarry)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env) # record episode statistics
   
    if create_video:
        env = gymnasium.wrappers.RecordVideo(env, 'video')

    # run the agent
    for i in range(NUM_EPISODES):
        done = False
        episode_steps = 0
        total_reward = 0
        obs, _ = env.reset()
        obs = convert_observation(obs)
        while not done:
            action = agent.get_action(obs)
            print(f"Current Action: {action}")
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            obs = convert_observation(obs)
            episode_steps += 1
            total_reward += reward

            # check if done
            if done:
                # extract the episode info
                cum_reward, episode_length, episode_time = info['episode']['r'][0], info['episode']['l'][0], info['episode']['t'][0]
                print(f"[{i} / {NUM_EPISODES}]\tEpisode Length: {episode_length}\t Cumulative Reward: {cum_reward}\t Episode Time: {episode_time}")
                break

            # print progress
            print(f"[{i} / {NUM_EPISODES}]\tEpisode Length: {episode_steps}\tReward: {reward}\tTotal Reward: {total_reward}\tAction: {action}", end="\r")

    # close the environment
    env.close()
