
# COPY THIS ENTIRE CELL INTO YOUR COLAB NOTEBOOK AND RUN IT
# This diagnostic script will:
# 1. Initialize the environment with the CURRENT configuration
# 2. Reset the robot
# 3. Force a STRAIGHT FORWARD command (1.0 m/s) ignoring the policy
# 4. Measure the actual velocity of each wheel and the body
# 5. Determine if wheels are fighting each other or if the robot is naturally spinning

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from brax.io import html
from IPython.display import HTML, display

print("--- STARTING DIAGNOSTIC RUN ---")

# 1. Setup Environment
# Ensure we use the latest CONFIG
try:
    # reconstructing env_kwargs from CONFIG if locally available
    # Assuming env_name and CONFIG exist in the notebook namespace
    diag_env_kwargs = {
        'reward_config': reward_config,
        'action_scale': 20.0, # Will be overridden by env but good to correspond
        'observation_history': policy_config.observation_history,
        # Add other args if your Env requires them, usually they are defaults or in CONFIG
    }
    # Better: reuse existing env_kwargs if possible
    if 'env_kwargs' in globals():
        print("Using existing env_kwargs from notebook...")
        diag_env = envs.get_environment(env_name, **env_kwargs)
    else:
        print("env_kwargs not found, attempting to create from CONFIG...")
        # This might fail if we miss parameters, but worth a try
        diag_env = envs.get_environment(env_name, **diag_env_kwargs)
        
    print("Environment created successfully.")
except Exception as e:
    print(f"Failed to create environment: {e}")
    print("Please ensure you have run the 'Configuration' cells in the notebook.")
    # Stop execution if env fails
    raise e

# 2. JIT functions
print("JIT-compiling step functions...")
diag_step = jax.jit(diag_env.step)
diag_reset = jax.jit(diag_env.reset)

# 3. Run Episode
rng = jax.random.PRNGKey(42)
state = diag_reset(rng)

# Force Command: [100% X Velocity, 0 Y, 0 Yaw]
# Scale typically ranges [-1, 1] mapped to linear velocity range
# But here command IS the target velocity?
# environment.py sample_command samples in m/s.
# So we set target to 1.0 m/s.
target_cmd = jp.array([1.0, 0.0, 0.0])
state.info['command'] = target_cmd

history = {
    'qvel_fr': [], 'qvel_fl': [], 'qvel_br': [], 'qvel_bl': [],
    'yaw_rate': [],
    'x_pos': [], 'y_pos': [],
    'rewards': []
}

states = []

print("Simulating 100 steps with FORCED FORWARD ACTION...")
# We will override the POLICY action and send a raw motor action
# Action 1.0 -> Max velocity (scaled by 25.0 -> 25 rad/s)
# Leg motors (indices 0,1, 3,4, etc) set to 0.0 (default pose)
# Wheel motors (indices 2, 5, 8, 11) set to +1.0
raw_action = np.zeros(12)
raw_action[[2, 5, 8, 11]] = 0.5 # Forward? +0.5 * 25 = 12.5 rad/s
jax_action = jp.array(raw_action)

for i in range(100):
    # Step with fixed action
    state = diag_step(state, jax_action)
    
    # Force command to remain straight (prevent resampling logic from changing it)
    state.info['command'] = target_cmd
    
    states.append(state)
    
    # Log Data
    # Wheel velocities from pipeline state
    # qvel has 6 root + 12 joints. Joint 0 is at index 6.
    # Wheel joints are at indices 2, 5, 8, 11 relative to start of joints
    # So indices 8, 11, 14, 17 in full qvel
    qv = state.pipeline_state.qvel
    history['qvel_fr'].append(qv[8])
    history['qvel_fl'].append(qv[11])
    history['qvel_br'].append(qv[14])
    history['qvel_bl'].append(qv[17])
    
    # Body Yaw Rate (approx qvel[5])
    history['yaw_rate'].append(qv[5])
    
    # Position
    pos = state.pipeline_state.x.pos[0] # Base body index 0
    history['x_pos'].append(pos[0])
    history['y_pos'].append(pos[1])
    
    # Rewards (sum of all terms)
    if 'rewards' in state.info:
        r_sum = sum(state.info['rewards'].values())
        history['rewards'].append(r_sum)

# 4. Analysis
print("\n--- DIAGNOSTIC RESULTS ---")
avg_fr = np.mean(history['qvel_fr'])
avg_fl = np.mean(history['qvel_fl'])
avg_br = np.mean(history['qvel_br'])
avg_bl = np.mean(history['qvel_bl'])

print(f"Average Wheel Velocities (rad/s):")
print(f"  Front Right: {avg_fr:.2f}")
print(f"  Front Left:  {avg_fl:.2f}")
print(f"  Back Right:  {avg_br:.2f}")
print(f"  Back Left:   {avg_bl:.2f}")

print(f"\nDisplacement: X={history['x_pos'][-1]:.2f}m, Y={history['y_pos'][-1]:.2f}m")
print(f"Average Yaw Rate: {np.mean(history['yaw_rate']):.3f} rad/s")

# Interpretation
if abs(avg_fr) < 0.1:
    print("\n[!] WARNING: Wheels are not moving under command! Check Action Scale or Stiffness.")
elif np.sign(avg_fr) != np.sign(avg_fl):
    print("\n[!!!] CRITICAL FAILURE: Left and Right wheels are rotating in OPPOSITE directions!")
    print("      This causes spinning. The XML axis or Quaternions need correction.")
elif np.sign(avg_fr) < 0:
    print("\n[!] NOTICE: Wheels are rotating NEGATIVE. Ideally positive command = forward.")
    print("    If displacement is negative X, motors are reversed relative to Convention.")
else:
    print("\n[OK] Wheels are rotating together in positive direction.")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['qvel_fr'], label='FR')
ax1.plot(history['qvel_fl'], label='FL')
ax1.plot(history['qvel_br'], label='BR')
ax1.plot(history['qvel_bl'], label='BL')
ax1.set_title("Wheel Velocities")
ax1.legend()
ax1.grid(True)

ax2.plot(history['x_pos'], history['y_pos'], '-o')
ax2.set_title("Robot Path (X vs Y)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.grid(True)
ax2.axis('equal')

plt.show()

# Render Video
print("Rendering debug video...")
display(HTML(html.render(diag_env.sys.tree_replace({'opt.timestep': 0.004}), 
                         [s.pipeline_state for s in states], height=320)))
