from typing import NamedTuple, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from MuZero.utils import logits_to_transformed_expected_value, normalize_hidden_state



class NetworkOutputs(NamedTuple):
    hidden_state: torch.Tensor
    reward: torch.Tensor
    pi_probs: torch.Tensor
    value: torch.Tensor

def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

            # nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            # nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)




class MuZeroNet(nn.Module):
    """Base MuZero model classs."""

    def __init__(
        self,
        num_actions: int,
        value_support_size: int = 31,
        reward_support_size: int = 31,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size

    @torch.no_grad()
    def initial_inference(self, x: torch.Tensor) -> NetworkOutputs:
        """During self-play, given environment observation, use representation function to predict initial hidden state.
        Then use prediction function to predict policy probabilities and state value (on the hidden state).
        The state value is a scalar."""
        # Representation function
        hidden_state = self.represent(x)

        # Prediction function
        pi_logits, value = self.prediction(hidden_state)
        pi_probs = F.softmax(pi_logits, dim=1)

        if not self.mse_loss_for_value:
            value = logits_to_transformed_expected_value(value, self.value_support_size)
        reward = torch.zeros_like(value)

        # Remove batch dimensions and turn into numpy or scalar values.
        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        reward = reward.squeeze(0).cpu().item()
        hidden_state = hidden_state.squeeze(0).cpu().numpy()

        return NetworkOutputs(hidden_state=hidden_state, reward=reward, pi_probs=pi_probs, value=value)

    @torch.no_grad()
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutputs:
        """During self-play, given hidden state at timestep `t-1` and action at t,
        use dynamics function to predict the reward and next hidden state,
        and use prediction function to predict policy probabilities and state value (on new hidden state).
        The reward and state value are scalars."""
        # Dynamics function
        hidden_state, reward = self.dynamics(hidden_state, action)

        if not self.mse_loss_for_reward:
            reward = logits_to_transformed_expected_value(reward, self.reward_support_size)

        # Prediction function
        pi_logits, value = self.prediction(hidden_state)
        pi_probs = F.softmax(pi_logits, dim=1)

        if not self.mse_loss_for_value:
            value = logits_to_transformed_expected_value(value, self.value_support_size)

        # Remove batch dimensions and turn into numpy or scalar values.
        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        reward = reward.squeeze(0).cpu().item()
        hidden_state = hidden_state.squeeze(0).cpu().numpy()

        return NetworkOutputs(hidden_state=hidden_state, reward=reward, pi_probs=pi_probs, value=value)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """Given environment observation, using representation function to predict initial hidden state."""
        raise NotImplementedError

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Given hidden state and action, unroll dynamics function to predict the next hidden state and the reward."""
        raise NotImplementedError

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden state, using prediction function to predict policy probabilities and state value.
        The state value is raw logits (no scaling or transformation)."""
        raise NotImplementedError

    @property
    def mse_loss_for_value(self):
        """Should use MSE loss for state value."""
        return self.value_support_size == 1

    @property
    def mse_loss_for_reward(self):
        """Should use MSE loss for reward."""
        return self.reward_support_size == 1
    


class MuZeroAtariNet(MuZeroNet):  # pylint: disable=abstract-method
    """MuZero model for Atari games."""

    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        num_res_blocks: int = 16,
        num_planes: int = 256,
        value_support_size: int = 601,
        reward_support_size: int = 601,
    ) -> None:
        super().__init__(num_actions, value_support_size, reward_support_size)
        self.represent_net = RepresentationConvAtariNet(input_shape, num_planes)

        dynamics_input_shape = (num_planes + num_actions, 6, 6)
        self.dynamics_net = DynamicsConvNet(dynamics_input_shape, num_actions, num_res_blocks, num_planes, reward_support_size)

        prediction_input_shape = (num_planes, 6, 6)
        self.prediction_net = PredictionConvNet(
            prediction_input_shape, num_actions, num_res_blocks, num_planes, value_support_size
        )

        initialize_weights(self)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.represent_net(x)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state, reward_logits

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(hidden_state)
    

class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_planes: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_planes),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out
    
class RepresentationConvAtariNet(nn.Module):
    """Representation function with Conv2d NN for Atari games."""

    def __init__(
        self,
        input_shape: Tuple,
        num_planes: int,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        # output 48x48
        self.conv_1 = nn.Conv2d(in_channels=c, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)

        self.res_blocks_1 = nn.Sequential(*[ResNetBlock(128) for _ in range(2)])

        # Original paper uses 3 res-blocks
        num_res_blocks = 2

        # output 24x24
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=num_planes, kernel_size=3, stride=2, padding=1, bias=False)

        self.res_blocks_2 = nn.Sequential(*[ResNetBlock(num_planes) for _ in range(num_res_blocks)])

        # output 12x12
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.res_blocks_3 = nn.Sequential(*[ResNetBlock(num_planes) for _ in range(num_res_blocks)])

        # output 6x6
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, predict the hidden state."""
        x = F.relu(self.conv_1(x))
        x = self.res_blocks_1(x)
        x = F.relu(self.conv_2(x))
        x = self.res_blocks_2(x)
        x = self.avg_pool_1(x)
        x = self.res_blocks_3(x)
        hidden_state = self.avg_pool_2(x)
        return hidden_state
    

class DynamicsConvNet(nn.Module):
    """Dynamics function with Conv2d NN for Atari and board games."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int,
        num_planes: int,
        support_size: int,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        c, h, w = input_shape

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_planes),
            nn.ReLU(),
        )

        res_blocks = []
        for _ in range(num_res_block):
            res_block = ResNetBlock(num_planes)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)

        self.reward_head = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * h * w, support_size),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state and action, predict the state transition and reward."""

        assert hidden_state.shape[0] == action.shape[0]

        b, c, h, w = hidden_state.shape

        # [batch_size, num_actions]
        onehot_action = F.one_hot(action.long(), self.num_actions).to(device=hidden_state.device, dtype=torch.float32)
        # [batch_size, num_actions, h*w]
        onehot_action = torch.repeat_interleave(onehot_action, repeats=h * w, dim=1)
        # [batch_size, num_actions, h, w]
        onehot_action = torch.reshape(onehot_action, (b, self.num_actions, h, w))

        x = torch.cat([hidden_state, onehot_action], dim=1)
        hidden_state = self.res_blocks(self.conv_block(x))
        reward_logits = self.reward_head(hidden_state)
        return hidden_state, reward_logits
    

class PredictionConvNet(nn.Module):
    """Prediction function with Conv2d NN for Atari and board games."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int,
        num_planes: int,
        support_size: int,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        res_blocks = []
        for _ in range(num_res_block):
            res_block = ResNetBlock(num_planes)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_net = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * h * w, num_actions),
        )

        self.value_net = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * h * w, support_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden state, predict the action probability and state value."""

        features = self.res_blocks(hidden_state)
        # Predict action distributions wrt policy
        pi_logits = self.policy_net(features)

        # Predict winning probability for current player's perspective.
        value_logits = self.value_net(features)

        return pi_logits, value_logits