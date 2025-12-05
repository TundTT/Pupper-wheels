
import mujoco
import time
import os
import numpy as np

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust path to where the XML is relative to this script
# Assuming I write this to: c:\Users\tundt\Desktop\Robotics\Pupper\Pupper wheel\Pupper-wheels\debug_spin.py
xml_path = os.path.join(script_dir, "description", "mujoco_xml", "Pupper_wheel_site_velocity_track.xml")

print(f"Loading model from: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Wheel indices based on environment.py (2, 5, 8, 11 in the actuator array?)
# Let's verify actuator names
wheel_names = ["wheel_front_r", "wheel_front_l", "wheel_back_r", "wheel_back_l"]
wheel_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in wheel_names]
print(f"Wheel Actuator IDs: {wheel_act_ids}")

# Simulation loop
print("Simulating for 2 seconds with constant +5.0 command to all wheels...")
print("The robot should move FORWARD. If it spins, one side is inverted.")

# Reset
mujoco.mj_resetData(model, data)
# Set initial pose to prevent immediate flop, but the xml has a keyframe 'home'
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)

max_steps = int(2.0 / model.opt.timestep)
positions_x = []
headings = []

for i in range(max_steps):
    # Command all wheels forward
    for act_id in wheel_act_ids:
        data.ctrl[act_id] = 5.0
        
    mujoco.mj_step(model, data)
    
    # Record position and heading
    positions_x.append(data.qpos[0]) # x position
    
    # Heading (quaternion to yaw)
    # qpos[3:7] is quat (w, x, y, z)
    q = data.qpos[3:7]
    # Yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    yaw = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    headings.append(yaw)

print(f"Final X Position: {positions_x[-1]:.3f} (Should be > 0)")
print(f"Final Yaw: {headings[-1]:.3f} (Should be near 0)")

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(positions_x)
plt.title("X Position")
plt.subplot(2,1,2)
plt.plot(headings)
plt.title("Yaw (Heading)")
plt.savefig("debug_spin_plot.png")
print("Saved debug_spin_plot.png")
