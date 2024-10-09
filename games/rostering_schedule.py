import datetime
import pathlib

import numpy as np
import torch

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # Basic settings
        self.seed = 0
        self.max_num_gpus = None
        
        # Fixed observation shape: 5 nurses, 4 shifts per day, 7 days in a week
        self.observation_shape = (5, 4, 7)
        # Action space: 28 actions (4 shifts * 7 days = 28 possible shift assignments)
        self.action_space = list(range(140))  # Actions represent assigning nurses to specific shifts on specific days
        # Only one player (the scheduler), so this remains a single-player game
        self.players = list(range(1))  # Single-player game (you should only edit the length)
        # No stacked observations
        self.stacked_observations = 0  # No need for previous observations/actions in this game

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.4
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper for nurse rostering.
    """

    def __init__(self, seed=None):
        self.env = NurseRostering()

class Game:
    """
    Game wrapper for nurse rostering.
    """

    def __init__(self, seed=None):
        self.env = NurseRostering()

    def step(self, action):
        """
        Apply the action (nurse assignment) to the game.
        """
        # Decode the action into nurse, shift, and day
        nurse = action // (4 * 7)  # Get the nurse index
        shift = (action % (4 * 7)) // 7  # Get the shift index
        day = action % 7  # Get the day index

        # Apply the action to the environment (assign nurse to shift on a day)
        observation, reward, done = self.env.step(nurse, day, shift)

        # Return observation, scaled reward, and done status
        return observation, reward * 20, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def expert_agent(self):
        return self.env.expert_action()

    def action_to_string(self, action_number):
        nurse = action_number // (4 * 7) + 1  # Add 1 to convert to 1-based index
        shift = (action_number % (4 * 7)) // 7  # Get the shift index
        day = action_number % 7  # Get the day index
        shift_name = ["Morning", "Evening", "Night", "Off"][shift]  # Shift names
        return f"Assign Nurse {nurse} to {shift_name} shift on Day {day + 1}"  # Day adjusted for 1-based index

class NurseRostering:
    def __init__(self):
        # Initialize a 5 x 4 x 7 schedule (nurses, shifts, days)
        # 0 means unassigned, 1 means assigned
        self.schedule = np.zeros((5, 4, 7), dtype="int32")  # (nurses, shifts, days)
        # Constraints like preferences, etc., can be added here
        self.preferences = self.initialize_preferences()

    def initialize_preferences(self):
        # Example: Random preference matrix for simplicity. You can adjust based on real preferences.
        return np.random.randint(0, 2, size=(5, 4, 7))  # 5 nurses, 4 shifts, 7 days

    def to_play(self):
        return 0  # There's only one player, the scheduler

    def reset(self):
        # Reset the schedule to an empty state
        self.schedule = np.zeros((5, 4, 7), dtype="int32")  # (nurses, shifts, days)
        return self.get_observation()

    def step(self, nurse, day, shift):
        # Apply the nurse assignment (nurse, shift, day)
        self.schedule[nurse, shift, day] = 1  # Assigning the shift on a particular day for the nurse

        # Check if scheduling is done (all shifts assigned)
        done = np.sum(self.schedule) == 5 * 4 * 7  # Every nurse should work 4 shifts/day for 7 days

        # Calculate reward based on preferences, coverage, etc.
        reward = self.calculate_reward(nurse, day, shift)

        return self.get_observation(), reward, done

    def get_observation(self):
        # Current schedule as the observation
        return self.schedule

    def legal_actions(self):
        # Return all unassigned slots in the schedule as legal actions
        legal = []
        for nurse in range(5):  # 5 nurses
            for shift in range(4):  # 4 shifts
                for day in range(7):  # 7 days
                    if self.schedule[nurse, shift, day] == 0:  # Unassigned slot
                        # Action encoding in the order nurse, shift, day
                        legal.append(nurse * (4 * 7) + shift * 7 + day)
        return legal

    def calculate_reward(self, nurse, day, shift):
        # Simple reward based on preferences: 1 if the nurse prefers the shift, -1 otherwise
        return 1 if self.preferences[nurse, shift, day] == 1 else -1

    def expert_action(self):
        # Return a random legal action for simplicity (can add smarter logic later)
        return np.random.choice(self.legal_actions())



