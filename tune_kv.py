# !!! COPY ALL OF THIS INTO A NEW COLAB CELL !!!
# Comprehensive kv Tuning Script - Finds the OPTIMAL kv value

import os
import sys
import subprocess

# 1. SETUP
print("--- Checking Environment ---")
try:
    import mujoco
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mujoco"])
    import mujoco

if not os.path.exists("Pupper-wheels"):
    subprocess.check_call(["git", "clone", "https://github.com/TundTT/Pupper-wheels", "-b", "Vibes"])

print("--- Setup Complete ---")

import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTS ---
WHEEL_RADIUS = 0.04445  # meters (from XML)
TARGET_LIN_VEL = 1.5  # m/s - REAL APPLICATION TARGET
WHEEL_CMD_RAD_S = -TARGET_LIN_VEL / WHEEL_RADIUS  # ≈ -33.7 rad/s (negative = forward)
EXPECTED_LIN_VEL = TARGET_LIN_VEL

def find_optimal_kv(kv_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]):
    """Test multiple kv values and find the one with best response."""
    
    # Navigate to XML folder
    target_dir = "Pupper-wheels/description/mujoco_xml"
    if os.path.isdir(target_dir):
        os.chdir(target_dir)
    
    # USE THE CORRECT XML FILE
    xml_path = "Pupper_wheel_site_velocity_track.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found")
        return
    
    with open(xml_path, 'r') as f:
        base_xml = f.read()

    results = {}
    plt.figure(figsize=(12, 6))
    
    for kv in kv_values:
        print(f"Testing kv = {kv}...")
        
        # Inject kv value
        modified_xml = base_xml.replace('kv="1"', f'kv="{kv}"')
        
        try:
            model = mujoco.MjModel.from_xml_string(modified_xml)
            data = mujoco.MjData(model)
        except Exception as e:
            print(f"Failed: {e}")
            continue

        # Find wheel actuators
        wheel_names = ["wheel_front_r", "wheel_front_l", "wheel_back_r", "wheel_back_l"]
        wheel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in wheel_names]
        wheel_ids = [i for i in wheel_ids if i != -1]

        # Simulation
        velocities = []
        timesteps = []
        duration = 3.0
        steps = int(duration / model.opt.timestep)
        
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.15  # Start on ground
        
        for _ in range(steps):
            data.ctrl[wheel_ids] = WHEEL_CMD_RAD_S
            mujoco.mj_step(model, data)
            velocities.append(data.qvel[0])  # X linear velocity
            timesteps.append(data.time)

        # --- METRICS ---
        velocities = np.array(velocities)
        steady_state_vel = np.mean(velocities[-100:])  # Last 0.4s
        
        # Rise time: time to reach 90% of steady state
        target_90 = 0.9 * steady_state_vel
        rise_time_idx = np.argmax(velocities >= target_90) if np.any(velocities >= target_90) else -1
        rise_time = timesteps[rise_time_idx] if rise_time_idx > 0 else float('inf')
        
        # Steady state error
        ss_error = abs(EXPECTED_LIN_VEL - steady_state_vel)
        
        results[kv] = {
            'steady_state': steady_state_vel,
            'rise_time': rise_time,
            'ss_error': ss_error,
            'score': steady_state_vel / (rise_time + 0.1) - ss_error  # Higher is better
        }
        
        plt.plot(timesteps, velocities, label=f'kv={kv}')

    # Plot expected velocity
    plt.axhline(y=EXPECTED_LIN_VEL, color='k', linestyle='--', label=f'Expected ({EXPECTED_LIN_VEL:.2f} m/s)')
    plt.title("kv Tuning: Robot Linear Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Linear Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print Results Table
    print("\n" + "="*70)
    print(f"{'kv':<10} | {'Steady State (m/s)':<20} | {'Rise Time (s)':<15} | {'Error (m/s)':<12}")
    print("="*70)
    for kv, metrics in sorted(results.items()):
        print(f"{kv:<10} | {metrics['steady_state']:<20.4f} | {metrics['rise_time']:<15.4f} | {metrics['ss_error']:<12.4f}")
    print("="*70)
    
    # Find best kv
    best_kv = max(results, key=lambda k: results[k]['score'])
    print(f"\n>>> OPTIMAL kv = {best_kv} <<<")
    print(f"    (Steady State: {results[best_kv]['steady_state']:.4f} m/s, Rise Time: {results[best_kv]['rise_time']:.4f} s)")

# --- RUN ---
find_optimal_kv()
