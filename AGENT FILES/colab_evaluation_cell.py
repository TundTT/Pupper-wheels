# --- Post-Training Evaluation Cell ---
# This script evaluates the trained policy and generates a video.
# It is optimized for GPU execution using JAX JIT compilation.

import jax
import jax.numpy as jp
import mediapy as media
import os
import time

print("Starting Post-Training Evaluation...")

# 1. Verify GPU availability
try:
    # Explicitly ask for GPU
    jax.config.update("jax_platform_name", "gpu")
    
    devices = jax.devices()
    print(f"JAX is running on: {devices}")
    
    # Check for GPU
    gpu_available = any("gpu" in str(d).lower() for d in devices)
    
    if not gpu_available:
        print("WARNING: JAX is NOT using a GPU. Evaluation will be slow.")
        print("Debug Info: Available devices:", jax.devices())
        print("If you are on Colab, go to Runtime > Change runtime type > Hardware accelerator > GPU.")
    else:
        print("SUCCESS: JAX is using GPU.")
        # Print detailed device info to confirm A100
        for i, device in enumerate(devices):
            print(f"Device {i}: {device.device_kind} (Platform: {device.platform})")
            
except Exception as e:
    print(f"Could not verify JAX devices: {e}")
    # Fallback to default backend if explicit GPU fails
    print("Falling back to default backend...")
    try:
        print(f"Default backend devices: {jax.devices()}")
    except:
        pass

try:
    # 2. Check for required variables from training session
    if 'eval_env' not in locals() or 'inference_fn' not in locals():
        # Try to load them if they are missing (fallback for fresh session)
        # This assumes standard naming from the notebook
        print("Variables 'eval_env' or 'inference_fn' not found. Attempting to use 'env' and 'make_inference_fn'...")
        if 'env' in locals() and 'params' in locals() and 'make_inference_fn' in locals():
             eval_env = env
             inference_fn = make_inference_fn(params)
             print("Restored environment and inference function from training variables.")
        else:
             raise NameError("Required variables (eval_env, inference_fn) or (env, params, make_inference_fn) are not defined. Please run the training cells first.")

    # 3. JIT Compile Critical Functions for GPU Speed
    # Compiling these functions ensures they run entirely on the GPU
    print("JIT compiling environment and policy functions...")
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(inference_fn)

    # 4. Initialize Simulation
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    # 5. Run Evaluation Loop
    # We use a Python loop here to collect states for rendering. 
    # Since the step function is JIT-compiled, the physics runs fast on GPU.
    rollout = []
    total_reward = 0
    steps = 500  # Duration of evaluation
    
    print(f"Running evaluation for {steps} steps...")
    start_time = time.time()
    
    for _ in range(steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        total_reward += state.reward
        
    duration = time.time() - start_time
    print(f"Evaluation completed in {duration:.2f} seconds ({steps/duration:.0f} steps/sec)")
    print(f"Total Reward: {total_reward:.4f}")

    # 6. Render and Save Video
    print("Rendering video (this may take a moment)...")
    # Using the tracking camera as in the training visualization
    video = eval_env.render(rollout, camera="tracking_cam")
    
    # Ensure output folder exists
    if 'output_folder' not in locals():
        output_folder = "."
        
    video_filename = os.path.join(output_folder, "post_training_eval.mp4")
    media.write_video(video_filename, video, fps=1.0 / eval_env.dt)
    print(f"Evaluation video saved to: {video_filename}")
    
    # 7. Save Metrics to File (for AI Analysis)
    metrics_path = os.path.join(output_folder, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Total Reward: {total_reward}\n")
        f.write(f"Evaluation Duration: {duration:.2f}s\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Steps per Second: {steps/duration:.2f}\n")
    print(f"Metrics saved to: {metrics_path}")

    # 8. Zip Results for Easy Download
    # Create a zip file containing the video and metrics
    zip_filename = os.path.join(output_folder, "evaluation_results")
    import shutil
    # We zip specific files to avoid zipping the whole folder if it's cluttered
    from zipfile import ZipFile
    with ZipFile(f"{zip_filename}.zip", 'w') as zip_obj:
        zip_obj.write(video_filename, os.path.basename(video_filename))
        zip_obj.write(metrics_path, os.path.basename(metrics_path))
        
    print(f"\nSUCCESS! Results zipped to: {zip_filename}.zip")
    print("Please download 'evaluation_results.zip' and extract it in your local workspace so we can analyze it together.")

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    import traceback
    traceback.print_exc()
