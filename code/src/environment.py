import gymnasium as gym
import numpy as np


# Inverted Pendulum Environment
class InvertedPendulumEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make("Pendulum-v1", max_episode_steps=300)
        super().__init__(env)
        self.unwrapped.max_torque = 15.0
        self.unwrapped.max_speed = 60.0
        self.unwrapped.action_space = gym.spaces.Box(
            low=-self.unwrapped.max_torque, high=self.unwrapped.max_torque, shape=(1,)
        )
        high = np.array([1.0, 1.0, self.unwrapped.max_speed])
        self.unwrapped.observation_space = gym.spaces.Box(low=-high, high=high)
        self.action_space = self.unwrapped.action_space
        self.observation_space = self.unwrapped.observation_space

    def reset(self, **kwargs):
        while True:
            obs, info = self.env.reset(**kwargs)
            if abs(self.unwrapped.state[0]) <= 1.0:
                return obs, info
            kwargs.pop("seed", None)


# Rigister Environment
def register_env():
    if "InvertedPendulum-v0" not in gym.envs.registry:
        gym.register(
            id="InvertedPendulum-v0", entry_point=lambda: InvertedPendulumEnv()
        )


# Build Environment
def make_env(env_id, idx, capture_video, run_name, gamma, normalize=True):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )

        return env

    return thunk
