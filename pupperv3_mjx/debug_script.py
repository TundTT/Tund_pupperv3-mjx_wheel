import jax
import jax.numpy as jp
from brax import envs
from pupperv3_mjx import environment
from pupperv3_mjx.debug_utils import run_debug_episode, print_reward_summary, plot_debug_metrics
import importlib

def run_sanity_check(env_kwargs):
    """
    Runs a sanity check on the PupperV3 environment.
    
    Args:
        env_kwargs: Dictionary of environment keyword arguments.
    """
    print("Reloading environment module...")
    importlib.reload(environment)

    print("Registering and creating environment...")
    if 'pupper' in envs._envs:
        del envs._envs['pupper']
    envs.register_environment('pupper', environment.PupperV3Env)
    
    # Create environment
    env = envs.get_environment('pupper', **env_kwargs)
    
    # Define Random Policy
    def random_inference_fn(params, obs, rng):
        action = jax.random.uniform(rng, (12,), minval=-1.0, maxval=1.0)
        return action, None

    # Run Debug Episode
    print("Running sanity check episode (200 steps)...")
    rewards_history, state_history = run_debug_episode(env, None, random_inference_fn, episode_length=200)

    # Print Summary
    print_reward_summary(rewards_history, state_history)

    # Plot Metrics
    plot_debug_metrics(rewards_history, state_history)
    
    return rewards_history, state_history
