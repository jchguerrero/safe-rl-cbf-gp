import copy as cpy
import random
import time

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from src.agent import Agent
from src.args import Args
from src.cbf import BarrierCompensator_nn, build_barrier, control_barrier
from src.dynamics_gp import (
    build_GP_model,
    get_GP_dynamics,
    gp_prediction_error,
    update_GP_dynamics,
)
from src.environment import make_env, register_env
from src.plot import plot_gp_error, plot_training
from torch.utils.tensorboard import SummaryWriter


def validate_and_derive_args(args: Args) -> Args:
    if args.num_envs != 1:
        raise ValueError(
            "this training script currently supports exactly one environment: --num-envs 1"
        )
    if args.num_steps <= 0:
        raise ValueError("num_steps must be greater than 0")
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be greater than 0")
    if args.num_minibatches <= 0:
        raise ValueError("num_minibatches must be greater than 0")
    if args.update_epochs <= 0:
        raise ValueError("update_epochs must be greater than 0")
    if args.bar_train_steps <= 0:
        raise ValueError("bar_train_steps must be greater than 0")

    args.batch_size = int(args.num_envs * args.num_steps)
    if args.total_timesteps < args.batch_size:
        raise ValueError("total_timesteps must be at least num_envs * num_steps")
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")

    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    return args


# Arguments
args = validate_and_derive_args(tyro.cli(Args))
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# CSV Logging setup
csv_path_episodes = f"runs/{run_name}/log_episodes.csv"
csv_path_iters = f"runs/{run_name}/log_iters.csv"

log_episodes = []
log_iters = []

# Register Environment
register_env()

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print(
    f"Starting training run '{run_name}' on {device} "
    f"for {args.total_timesteps} timesteps ({args.num_iterations} iterations)."
)

# Environment
envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            args.env_id,
            i,
            args.capture_video,
            run_name,
            args.gamma,
            normalize=args.normalize,
        )
        for i in range(args.num_envs)
    ]
)
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

# Agent and Optimizer
agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# CBF Parameters
torque_bound = 15.0
max_speed = 60.0
action_size = int(np.prod(envs.single_action_space.shape))
obs_size = int(np.array(envs.single_observation_space.shape).prod())
P, q, H1, H2, H3, H4, F = build_barrier(action_size)

# Barrier Compensator
bar_comp_nn = BarrierCompensator_nn(obs_size, action_size).to(device)
bar_comp_nn_optimizer = optim.Adam(bar_comp_nn.parameters(), lr=args.bar_lr)

# Build GP model of dynamics
GP_model = build_GP_model(obs_size)
GP_model_prev = None
firstIter = True

# Storage Setup
obs = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_observation_space.shape
).to(device)
actions = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_action_space.shape
).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
action_bar_arr = np.zeros((args.num_steps, action_size))
action_BAR_arr = np.zeros((args.num_steps, action_size))
episode_obs_list = []
episode_action_list = []

# For episode tracking
episode_theta_list = []
episode_ucbf_list = []
episode_safe_list = []
ep_max_angle = 0.0
ep_mean_ucbf = 0.0
ep_max_ucbf = 0.0
ep_violations = 0

# Initialization
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(args.num_envs).to(device)

# Policy Training
for iteration in range(1, args.num_iterations + 1):
    # Learning rate Annealing
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    # GP first iteration check
    if not firstIter:
        GP_model_prev = cpy.copy(GP_model)
        GP_model = build_GP_model(obs_size)
    timesteps_in_batch = 0

    # One-step error accumulator
    iter_errs = []

    ###########################################
    # Rollout and collect data
    ###########################################
    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        ###########################################
        # RL-CBF-GP guided
        ###########################################
        # Sample action from RL policy
        with torch.no_grad():
            action_rl, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action_rl
        logprobs[step] = logprob

        # CBF-Neural-Network output
        prev_obs_expanded = np.expand_dims(np.squeeze(next_obs[0].cpu().numpy()), 0)
        u_BAR_ = bar_comp_nn.get_action(prev_obs_expanded)

        # RL control compensated by the CBF-Neural-Network output
        u_RL = float(action_rl.cpu().numpy()[0, 0])
        u_RL_comp = u_RL + float(u_BAR_[0, 0])

        # Dynamics with GP included
        if firstIter:
            [f, g, x, std] = get_GP_dynamics(GP_model, prev_obs_expanded, u_RL_comp)
        else:
            [f, g, x, std] = get_GP_dynamics(
                GP_model_prev, prev_obs_expanded, u_RL_comp
            )

        # Apply CBF correction (filter)
        u_bar_ = control_barrier(
            u_RL_comp, f, g, x, std, P, q, H1, H2, H3, H4, F, torque_bound, max_speed
        )
        u_k = u_RL_comp + u_bar_[0]
        u_k_np = np.array([[u_k]], dtype=np.float32)

        # Saving data for logs
        theta_now = x[0]
        omega_now = x[1]
        episode_theta_list.append(abs(np.degrees(theta_now)))
        episode_ucbf_list.append(abs(float(u_bar_[0])))
        episode_safe_list.append(int(abs(theta_now) + 0.01 * abs(omega_now) > F))

        # Store for compensator training
        action_bar_arr[step] = u_bar_
        action_BAR_arr[step] = u_BAR_[0]
        episode_obs_list.append(np.squeeze(next_obs[0].cpu().numpy()))
        episode_action_list.append(u_k)

        # Agent step with CBF-corrected action
        next_obs, reward, terminations, truncations, infos = envs.step(u_k_np)
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done
        ).to(device)

        # End of Episode
        if terminations[0] or truncations[0]:
            if len(episode_obs_list) > 1:
                episode_obs_array = np.array(episode_obs_list)
                episode_action_array = np.array(episode_action_list)

                # GP update
                if timesteps_in_batch < args.timestep_batch_GP_max:
                    update_GP_dynamics(
                        GP_model, episode_obs_array, episode_action_array, obs_size
                    )
                    errs = gp_prediction_error(
                        GP_model, episode_obs_array, episode_action_array, obs_size
                    )
                    if errs is not None:
                        iter_errs.append(errs)

            ep_max_angle = max(episode_theta_list) if episode_theta_list else 0.0
            ep_mean_ucbf = (
                float(np.mean(episode_ucbf_list)) if episode_ucbf_list else 0.0
            )
            ep_max_ucbf = float(np.max(episode_ucbf_list)) if episode_ucbf_list else 0.0
            ep_violations = int(sum(episode_safe_list)) if episode_safe_list else 0

            timesteps_in_batch += len(episode_obs_list)
            episode_obs_list = []
            episode_action_list = []

            episode_theta_list = []
            episode_ucbf_list = []
            episode_safe_list = []

        # Logging
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

                    # CSV logs
                    log_episodes.append(
                        {
                            "global_step": global_step,
                            "iteration": iteration,
                            "episode_return": round(
                                float(info["episode"]["r"].item()), 4
                            ),
                            "episode_length": int(info["episode"]["l"].item()),
                            "max_angle": round(ep_max_angle, 3),
                            "safety_violations": ep_violations,
                            "mean_u_cbf": round(ep_mean_ucbf, 6),
                            "max_u_cbf": round(ep_max_ucbf, 4),
                        }
                    )

    firstIter = False

    ###########################################
    # Train CBF Compensator Neural Network
    ###########################################
    # Observation and Target
    obs_udeployed = obs.reshape((-1,) + envs.single_observation_space.shape).detach()
    correction_cbf2_target = torch.FloatTensor(action_bar_arr + action_BAR_arr).to(
        device
    )

    # Training
    for _ in range(args.bar_train_steps):
        correction_deployed = bar_comp_nn(obs_udeployed)
        comp_loss = nn.functional.mse_loss(correction_deployed, correction_cbf2_target)
        bar_comp_nn_optimizer.zero_grad()
        comp_loss.backward()
        bar_comp_nn_optimizer.step()

    writer.add_scalar("losses/compensator_loss", comp_loss.item(), global_step)

    ###########################################
    # Compute GAE
    ###########################################
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

    # Flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    ###########################################
    # Optimizing the Policy and Value network
    ###########################################
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        # Batch iteration
        for start in range(0, args.batch_size, args.minibatch_size):
            # Minibatch
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            # Policies Ratio
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # KL divergence approx.
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                ]

            # Advantage normalization
            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss (Clipped PPO)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Entropy
            entropy_loss = entropy.mean()

            # Total loss
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # Early stopping
        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    # Logging
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # csv log
    u_cbf_batch = np.abs(action_bar_arr)
    mean_u_cbf_batch = float(u_cbf_batch.mean())
    max_u_cbf_batch = float(u_cbf_batch.max())
    cbf_active_frac = float((u_cbf_batch > 1e-4).mean())
    writer.add_scalar("safety/mean_u_cbf", mean_u_cbf_batch, global_step)
    writer.add_scalar("safety/cbf_active_frac", cbf_active_frac, global_step)

    # GP prediction error
    keys = ["nom_rmse", "gp_rmse", "nom_rmse_large", "gp_rmse_large"]
    err_means = {
        k: (float(np.nanmean([e[k] for e in iter_errs])) if iter_errs else np.nan)
        for k in keys
    }
    for k in keys:
        writer.add_scalar(f"gp/{k}", err_means[k], global_step)

    log_iters.append(
        {
            "global_step": global_step,
            "iteration": iteration,
            "policy_loss": round(pg_loss.item(), 6),
            "value_loss": round(v_loss.item(), 6),
            "compensator_loss": round(comp_loss.item(), 6),
            "mean_u_cbf_batch": round(mean_u_cbf_batch, 6),
            "max_u_cbf_batch": round(max_u_cbf_batch, 4),
            "cbf_active_frac": round(cbf_active_frac, 4),
            "explained_variance": round(explained_var, 4),
            "approx_kl": round(approx_kl.item(), 6),
            **{k: round(err_means[k], 6) for k in keys},
        }
    )

    # Record rewards
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar(
        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )

if args.save_model:
    model_path = f"runs/{run_name}/{args.exp_name}.pt"
    gp_path = f"runs/{run_name}/{args.exp_name}_gp.joblib"

    # Save PPO Policy and CBF NN Compensator
    torch.save(
        {
            "agent": agent.state_dict(),
            "bar_comp_nn": bar_comp_nn.state_dict(),
        },
        model_path,
    )

    # Save GP model
    joblib.dump(GP_model, gp_path)
    print(f"Models saved to runs/{run_name}/")
else:
    print("Model saving disabled.")

# Save logs
pd.DataFrame(log_episodes).to_csv(csv_path_episodes, index=False)
pd.DataFrame(log_iters).to_csv(csv_path_iters, index=False)
print(f"Logs saved to runs/{run_name}/")

envs.close()
writer.close()

plot_training(
    episodes_path=csv_path_episodes,
    iters_path=csv_path_iters,
    output_path=f"runs/{run_name}/training_curves.png",
    F=1.0,
)

plot_gp_error(
    iters_path=csv_path_iters,
    output_path=f"runs/{run_name}/gp_error.png",
)
