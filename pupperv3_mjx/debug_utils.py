import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks

def run_debug_episode(env, params, inference_fn, rng_seed=0, episode_length=500):
    """
    Runs a single episode and collects detailed metrics and state data.
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
    
    for _ in range(episode_length):
        step_rng, action_rng = jax.random.split(step_rng)
        
        # Get action from policy
        action, _ = inference_fn(params, state.obs, action_rng)
        
        # Step environment
        state = env.step(state, action)
        
        # Collect Metrics
        for k, v in state.info['rewards'].items():
            rewards_history[k].append(float(v))
            
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
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
            
        state_history['z_height'].append(float(pos[2]))
        state_history['roll'].append(float(roll))
        state_history['pitch'].append(float(pitch))
        state_history['action_mag'].append(float(jp.linalg.norm(action)))
        
        # Velocity
        vel = state.pipeline_state.xd.vel[0]
        ang = state.pipeline_state.xd.ang[0]
        state_history['linear_vel'].append(float(jp.linalg.norm(vel)))
        state_history['angular_vel'].append(float(jp.linalg.norm(ang)))

        if state.done:
            break
            
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

def print_reward_summary(rewards_history):
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
