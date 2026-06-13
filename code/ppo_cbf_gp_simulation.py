from argparse import ArgumentParser
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_PATH = PROJECT_ROOT / "results" / "trained_models" / "inverted_pendulum.pt"
DEFAULT_GP_PATH = (
    PROJECT_ROOT / "results" / "trained_models" / "inverted_pendulum_gp.joblib"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "pendulum_simulation.mp4"
ENV_ID = "InvertedPendulum-v0"
RUN_NAME = "eval"
SEED = 1
CUDA = False
GAMMA = 0.995
NORMALIZE = False
CAPTURE_VIDEO = False
MAX_STEPS = 300
TORQUE_BOUND = 15.0
MAX_SPEED = 60.0
SAFE_LIMIT_RAD = 1.0


def parse_args():
    parser = ArgumentParser(
        description="Run one PPO-CBF-GP inverted pendulum simulation."
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=DEFAULT_CKPT_PATH,
        help="Path to the trained .pt checkpoint.",
    )
    parser.add_argument(
        "--gp-path",
        type=Path,
        default=DEFAULT_GP_PATH,
        help="Path to the trained GP .joblib model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the simulation MP4 will be saved.",
    )
    return parser.parse_args()


# Reset
def reset_safe(env, seed=None):
    while True:
        obs, info = env.reset(seed=seed)
        if abs(env.unwrapped.state[0]) <= SAFE_LIMIT_RAD:
            return obs, info
        seed = None


def save_video(frames, thetas, actions, output_path):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    img = ax.imshow(frames[0])
    title = ax.set_title("t=0.00s")

    def update(i):
        img.set_data(frames[i])
        theta = thetas[i] if i < len(thetas) else 0.0
        action = actions[i] if i < len(actions) else 0.0
        title.set_text(f"t={i * 0.05:.2f}s | theta={theta:.2f} | u={action:.2f}")
        return [img, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer="ffmpeg", fps=20, dpi=100)
    plt.close(fig)
    print(f"Saved video to: {output_path}")


def main():
    # Arguments
    args = parse_args()
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.gp_path.exists():
        raise FileNotFoundError(f"GP model not found: {args.gp_path}")

    import gymnasium as gym
    import joblib
    import numpy as np
    import torch
    from src.agent import Agent
    from src.cbf import BarrierCompensator_nn, build_barrier, control_barrier
    from src.dynamics_gp import get_GP_dynamics
    from src.environment import make_env, register_env

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")

    # Environment
    register_env()
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, 0, CAPTURE_VIDEO, RUN_NAME, GAMMA, normalize=NORMALIZE)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    action_size = int(np.prod(envs.single_action_space.shape))
    obs_size = int(np.array(envs.single_observation_space.shape).prod())

    # Load checkpoint and GP model
    agent = Agent(envs).to(device)
    bar_comp_nn = BarrierCompensator_nn(obs_size, action_size).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    bar_comp_nn.load_state_dict(checkpoint["bar_comp_nn"])
    gp_model = joblib.load(args.gp_path)

    # Evaluation mode
    agent.eval()
    bar_comp_nn.eval()

    # Build CBF
    p_mat, q_vec, h1, h2, h3, h4, safe_limit = build_barrier(action_size)

    # Environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    env.unwrapped.max_torque = TORQUE_BOUND
    env.unwrapped.max_speed = MAX_SPEED
    env.unwrapped.action_space = gym.spaces.Box(
        low=-env.unwrapped.max_torque,
        high=env.unwrapped.max_torque,
        shape=(1,),
    )
    high = np.array([1.0, 1.0, env.unwrapped.max_speed])
    env.unwrapped.observation_space = gym.spaces.Box(low=-high, high=high)

    # Simulation
    obs, _ = reset_safe(env, seed=SEED)
    frames, actions, thetas = [], [], []
    ep_return = 0.0
    max_angle = 0.0

    for _ in range(MAX_STEPS):
        obs_np = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        # RL control and CBF-nn compensator
        with torch.no_grad():
            u_rl = float(agent.actor_mean(obs_t).cpu().numpy()[0, 0])
            u_bar_nn = float(bar_comp_nn(obs_t).cpu().numpy()[0, 0])

        # Control: RL+CBF+GP
        u_rl_comp = u_rl + u_bar_nn
        f, g, x, std = get_GP_dynamics(gp_model, obs_np)
        u_bar = control_barrier(
            u_rl_comp,
            f,
            g,
            x,
            std,
            p_mat,
            q_vec,
            h1,
            h2,
            h3,
            h4,
            safe_limit,
            TORQUE_BOUND,
            MAX_SPEED,
        )
        u_k = u_rl_comp + u_bar[0]
        deployed_action = np.array(
            [np.clip(u_k, -TORQUE_BOUND, TORQUE_BOUND)], dtype=np.float32
        )

        # Step in environment
        obs, reward, terminated, truncated, _ = env.step(deployed_action)
        env.unwrapped.last_u = float(env.unwrapped.last_u)
        frames.append(env.render())

        theta = float(env.unwrapped.state[0])
        max_angle = max(max_angle, abs(theta))
        ep_return += float(reward)
        actions.append(float(deployed_action[0]))
        thetas.append(theta)

        if terminated or truncated:
            break

    env.close()
    envs.close()

    # Basic Metrics
    print(f"Episode return: {ep_return:.3f}")
    print(f"Max angle: {max_angle:.3f} rad")
    print(f"Safe: {max_angle <= SAFE_LIMIT_RAD}")
    print(f"Steps: {len(thetas)}")

    # Save simulation (video) in .mp4 format
    if frames:
        save_video(frames, thetas, actions, args.output)


if __name__ == "__main__":
    main()
