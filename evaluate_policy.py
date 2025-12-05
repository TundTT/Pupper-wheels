
import os
import sys
# Set environment variables before importing jax/brax
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'
# os.environ['MUJOCO_GL'] = 'egl' # Might be needed for headless rendering

import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
from datetime import datetime
from pathlib import Path
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from etils import epath

# Add pupperv3_mjx to path
sys.path.append(os.path.abspath("../Tund_pupperv3-mjx_wheel"))
from pupperv3_mjx import utils
from pupperv3_mjx import envs as pupper_envs

def evaluate(checkpoint_path, output_folder="evaluation_results"):
    """
    Evaluates a trained policy from a checkpoint.
    """
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load environment (configuration needs to match training)
    # Assuming default configuration for now, or we might need to load config from checkpoint if saved
    env_name = "PupperV3-Wheel" 
    # Register environment if not already registered (assuming pupper_envs does this or we need to do it)
    # In the notebook: envs.register_environment('pupperv3_wheel', ...)
    
    # We need to replicate the environment registration from the notebook
    # Let's assume the user will run this script in an environment where they can access the notebook code or we copy it.
    # For now, I will try to import the environment definition from the package if possible, 
    # or redefine it here based on the notebook content.
    
    # Re-registering environment as in the notebook
    from pupperv3_mjx.envs import pupper_v3_wheel_task
    envs.register_environment('pupperv3_wheel', pupper_v3_wheel_task.PupperV3WheelTask)

    env = envs.get_environment('pupperv3_wheel')
    
    # Load params
    # This part is tricky because orbax loading might require the original structure.
    # If the notebook saved it using model.save_params, we can use model.load_params
    try:
        params = model.load_params(checkpoint_path)
    except Exception as e:
        print(f"Error loading params with model.load_params: {e}")
        # Try orbax directly if needed, but model.load_params is standard for brax
        return

    # Define inference function
    # We need the make_policy function. It usually comes from ppo.train or ppo.make_inference_fn
    # But ppo.make_inference_fn requires the network definition.
    # Let's use ppo.make_inference_fn
    
    # We need to know the network architecture used in training.
    # Assuming default PPO networks.
    normalize = lambda x, y: x
    network_factory = ppo_networks.make_ppo_networks
    
    # Create the inference function
    make_inference_fn = ppo_networks.make_inference_fn(
        network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=normalize
        )
    )
    
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Run evaluation
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    # Rollout
    rollout = []
    total_reward = 0
    steps = 500 # Evaluate for 500 steps
    
    for _ in range(steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        total_reward += state.reward
        
    print(f"Total Reward: {total_reward}")
    
    # Visualize
    # Render video
    video_path = os.path.join(output_folder, "eval_video.mp4")
    # env.render might need specific arguments
    # In notebook: eval_env.render(rollout[::render_every], camera="tracking_cam")
    
    # We need to create an eval_env for rendering if the base env doesn't support it directly in the same way
    # But brax envs usually support render.
    
    try:
        video = env.render(rollout, camera="tracking_cam")
        media.write_video(video_path, video, fps=1.0 / env.dt)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Error rendering video: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_policy.py <checkpoint_path>")
    else:
        evaluate(sys.argv[1])
