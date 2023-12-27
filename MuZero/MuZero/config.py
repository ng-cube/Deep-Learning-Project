import collections
from typing import Callable, List, Optional 

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):
    """ MuZero configuration object. """
    def __init__(
        self,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        lr_init: float,
        lr_milestones: List[int],
        visit_softmax_temperature_fn: Callable[[int, int], float],
        known_bounds: Optional[KnownBounds] = None,
        num_training_steps: Optional[int] = int(1000e3),
        checkpoint_interval: Optional[int] = int(1e3),
        num_planes: Optional[int] = 256,
        num_res_blocks: Optional[int] = 16,
        hidden_dim: Optional[int] = 64,
        value_support_size: Optional[int] = 1,
        reward_support_size: Optional[int] = 1,
        train_delay: Optional[float] = 0.0,
        min_replay_size: Optional[int] = int(2e4),
        acc_seq_length: Optional[int] = int(200),
        clip_grad: Optional[bool] = False,
        use_tensorboard: Optional[bool] = False,
        is_board_game: Optional[bool] = False,
    ) -> None:

        # Network Architecture
        self.num_planes = num_planes
        self.num_res_blocks = num_res_blocks
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        self.hidden_dim = hidden_dim  # hidden state dimension for MLP network only

        # Self-Play
        # visit_softmax_temperature_fn takes in two parameters (env_steps, train_steps)
        self.visit_softmax_temperature_fn: Callable[[int, int], float] = visit_softmax_temperature_fn
        self.num_simulations = num_simulations
        self.discount = discount
        # Send samples to learning when self-play (measured in env steps) accumulated this sequence length.
        self.acc_seq_length = acc_seq_length

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_eps = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behavior to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Training
        self.num_training_steps = num_training_steps
        self.checkpoint_interval = checkpoint_interval
        # Unlike in the paper, replay size are measured by single sample, not an entire game.
        self.min_replay_size = min_replay_size

        self.batch_size = batch_size
        self.unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.clip_grad = clip_grad
        self.max_grad_norm = 40.0

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_milestones = lr_milestones

        self.use_tensorboard = use_tensorboard
        self.train_delay = train_delay
        self.is_board_game = is_board_game


def make_atari_config(
    num_training_steps: int = int(10e6),
    batch_size: int = 128,
    min_replay_size: int = 10000,
    use_tensorboard: bool = True,
    clip_grad: bool = False,
) -> MuZeroConfig:
    """ Returns MuZero config for openAI Gym Atari games. """
    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=batch_size,
        td_steps=10,
        lr_init=0.05,
        lr_milestones=[1e4, 2e4], #[100e3, 200e3],
        visit_softmax_temperature_fn=atari_visit_softmax_temperature_fn,
        num_training_steps=num_training_steps,
        num_planes=128,  # 256
        num_res_blocks=8,  # 16
        hidden_dim=0,
        value_support_size=61,  # 601
        reward_support_size=61,  # 601
        min_replay_size=min_replay_size,
        acc_seq_length=200,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=False,
    )



def atari_visit_softmax_temperature_fn(env_steps, training_steps):
    if training_steps < 1e4: ##500e3:
        return 1.0
    elif training_steps < 5e4: #1000e3:
        return 0.5
    else:
        return 0.25