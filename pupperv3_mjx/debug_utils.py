import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks

def run_debug_episode(env, params, inference_fn, rng_seed=0, episode_length=500, command_override=None):
    """
    Runs a single episode and collects detailed metrics and state data.
    
    Args:
        env: The environment.
        params: Policy parameters.
        inference_fn: Inference function.
        rng_seed: Random seed.
        episode_length: Length of the episode.
        command_override: Optional JAX array of shape (3,) [lin_vel_x, lin_vel_y, ang_vel_yaw].
                          If provided, this command will be forced at every step.
    """
    rng = jax.random.PRNGKey(rng_seed)
    reset_rng, step_rng = jax.random.split(rng)
    
    state = env.reset(reset_rng)
    
    # Initialize history lists
    rewards_history = {k: [] for k in state.info['rewards'].keys()}
    state_history = {
        'z_height': [],
        'roll': [],
        'pitch': [],
        'wheel_vel': [],
        'action_mag': [],
        'linear_vel': [],
        'angular_vel': []
    }
    
    # JIT compile functions for speed
    jit_step = jax.jit(env.step)
    if params is None:
        # Handle random policy case where params is None
        # We need to wrap inference_fn to handle None params if it doesn't already
        jit_inference_fn = jax.jit(lambda p, o, r: inference_fn(p, o, r))
    else:
        jit_inference_fn = jax.jit(inference_fn)

    for _ in range(episode_length):
        step_rng, action_rng = jax.random.split(step_rng)
        
        # Override command if provided
        if command_override is not None:
            # We need to update state.info['command']
            # Since state is a frozen struct, we use replace.
            # info is a dict, so we copy and update.
            new_info = state.info.copy()
            new_info['command'] = command_override
            state = state.replace(info=new_info)
        
        # Get action from policy
        action, _ = jit_inference_fn(params, state.obs, action_rng)
        
        # Step environment
        state = jit_step(state, action)
        
        # Collect Metrics
        for k, v in state.info['rewards'].items():
            rewards_history[k].append(v)
            
        # Collect State Data
        # Assuming standard PupperV3Env state structure
        # x.pos is (num_envs, 3), we take the first env
        pos = state.pipeline_state.x.pos[0] # Torso is usually index 0
        rot = state.pipeline_state.x.rot[0]
        
        # Convert quaternion to roll/pitch (approximate)
        # q = [w, x, y, z]
        q = rot
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        roll = jp.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        # Use jp.where for JAX compatibility instead of if/else
        pitch = jp.where(
            jp.abs(sinp) >= 1,
            jp.copysign(jp.pi / 2, sinp),
            jp.arcsin(sinp)
        )
            
        state_history['z_height'].append(pos[2])
        state_history['roll'].append(roll)
        state_history['pitch'].append(pitch)
        state_history['action_mag'].append(jp.linalg.norm(action))
        
        # Velocity
        vel = state.pipeline_state.xd.vel[0]
        ang = state.pipeline_state.xd.ang[0]
        state_history['linear_vel'].append(jp.linalg.norm(vel))
        state_history['angular_vel'].append(jp.linalg.norm(ang))

        if state.done:
            break
            
    # Convert history to numpy arrays at the end to minimize GPU-CPU sync
    for k in rewards_history:
        rewards_history[k] = np.array([float(x) for x in rewards_history[k]])
        
    for k in state_history:
        state_history[k] = np.array([float(x) for x in state_history[k]])
            
    return rewards_history, state_history

def plot_debug_metrics(rewards_history, state_history):
    """
    Plots the collected metrics.
    """
    # 1. Plot Rewards
    plt.figure(figsize=(15, 10))
    for k, v in rewards_history.items():
        if np.sum(np.abs(v)) > 1e-6: # Only plot non-zero rewards
            plt.plot(v, label=k)
    plt.title("Reward Components over Time")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Reward Value")
    plt.grid(True)
    plt.show()
    
    # 2. Plot State (Height, Roll, Pitch)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    axes[0].plot(state_history['z_height'])
    axes[0].set_title("Torso Z-Height")
    axes[0].set_ylabel("Meters")
    axes[0].grid(True)
    
    axes[1].plot(state_history['roll'], label='Roll')
    axes[1].plot(state_history['pitch'], label='Pitch')
    axes[1].set_title("Body Orientation")
    axes[1].set_ylabel("Radians")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(state_history['linear_vel'], label='Lin Vel')
    axes[2].plot(state_history['angular_vel'], label='Ang Vel')
    axes[2].set_title("Body Velocity Magnitude")
    axes[2].set_ylabel("m/s or rad/s")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.show()

def print_reward_summary(rewards_history, state_history):
    """
    Prints a summary of the rewards.
    """
    print("\n--- Reward Summary ---")
    print(f"{'Reward Name':<30} | {'Mean':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 70)
    for k, v in rewards_history.items():
        vals = np.array(v)
        if np.sum(np.abs(vals)) > 1e-6:
            print(f"{k:<30} | {np.mean(vals):.4f}     | {np.min(vals):.4f}     | {np.max(vals):.4f}")
    print("-" * 70)

    # Print State Summary
    print("\n--- State Summary ---")
    print(f"{'Metric':<30} | {'Mean':<10} | {'Max':<10}")
    print("-" * 70)
    for k in ['linear_vel', 'angular_vel', 'z_height']:
        if k in state_history:
            vals = state_history[k]
            print(f"{k:<30} | {np.mean(vals):.4f}     | {np.max(vals):.4f}")
    print("-" * 70)

def print_reward_distribution(rewards_history):
    """
    Prints a detailed distribution of rewards to help with tuning.
    Calculates the contribution of each term to the total reward.
    """
    print("\n=== Reward Distribution Analysis ===")
    
    # Calculate total reward per step
    total_rewards = np.zeros(len(next(iter(rewards_history.values()))))
    for v in rewards_history.values():
        total_rewards += v
        
    avg_total_reward = np.mean(total_rewards)
    print(f"Average Total Reward per Step: {avg_total_reward:.4f}")
    print("-" * 90)
    print(f"{'Reward Name':<30} | {'Mean':<10} | {'% of Total':<12} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 90)
    
    # Sort by absolute mean value to show most impactful terms first
    sorted_items = sorted(rewards_history.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)
    
    for k, v in sorted_items:
        vals = np.array(v)
        mean_val = np.mean(vals)
        if np.abs(mean_val) > 1e-6:
            # Avoid division by zero
            percent = (mean_val / avg_total_reward * 100) if abs(avg_total_reward) > 1e-6 else 0.0
            print(f"{k:<30} | {mean_val:.4f}     | {percent:>9.1f}%   | {np.std(vals):.4f}     | {np.min(vals):.4f}     | {np.max(vals):.4f}")
    print("-" * 90)
