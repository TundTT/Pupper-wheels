
import os
import sys
import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np
from ml_collections import config_dict
import importlib

print("--- APPLYING SPINNING FIX PATCH ---")

# 1. FIND LIBRARY LOCATION
# We will try to find where pupperv3_mjx is installed/located
try:
    import pupperv3_mjx
    lib_path = os.path.dirname(pupperv3_mjx.__file__)
    print(f"Found library at: {lib_path}")
except ImportError:
    # Fallback for Colab default path if not imported yet
    lib_path = "/content/pupperv3_mjx/pupperv3_mjx"
    if not os.path.exists(lib_path):
        # Fallback for local windows path structure seen in logs
        lib_path = "pupperv3_mjx"
    print(f"Library not imported, assuming path: {lib_path}")

if not os.path.exists(lib_path):
    print(f"ERROR: Could not find pupperv3_mjx directory at {lib_path}")
    print("Please run the repository installation cell first!")
else:
    # 2. OVERWRITE REWARDS.PY
    rewards_code = r'''
import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np

EPS = 1e-6

# ------------ reward functions----------------
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    return jp.clip(jp.square(xd.vel[0, 2]), -1000.0, 1000.0)

def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    return jp.clip(jp.sum(jp.square(xd.ang[0, :2])), -1000.0, 1000.0)

def reward_ang_vel_z(xd: Motion) -> jax.Array:
    # NEW: Quadratic penalty for spinning
    return jp.clip(jp.square(xd.ang[0, 2]), -1000.0, 1000.0)

def reward_tracking_orientation(desired_world_z_in_body_frame: jax.Array, x: Transform, tracking_sigma: float) -> jax.Array:
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    return jp.clip(jp.exp(-error / (tracking_sigma + EPS)), -1000.0, 1000.0)

def reward_orientation(x: Transform) -> jax.Array:
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.clip(jp.sum(jp.square(rot_up[:2])), -1000.0, 1000.0)

def reward_torques(torques: jax.Array) -> jax.Array:
    return jp.clip(jp.sum(jp.square(torques)), -1000.0, 1000.0)

def reward_joint_acceleration(joint_vel: jax.Array, last_joint_vel: jax.Array, dt: float) -> jax.Array:
    return jp.clip(jp.sum(jp.square((joint_vel - last_joint_vel) / (dt + EPS))), -1000.0, 1000.0)

def reward_mechanical_work(torques: jax.Array, velocities: jax.Array) -> jax.Array:
    return jp.clip(jp.sum(jp.abs(torques * velocities)), -1000.0, 1000.0)

def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.clip(jp.sum(jp.square(act - last_act)), -1000.0, 1000.0)

def reward_tracking_lin_vel(commands: jax.Array, x: Transform, xd: Motion, tracking_sigma) -> jax.Array:
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(-lin_vel_error / (tracking_sigma + EPS))
    return jp.clip(lin_vel_reward, -1000.0, 1000.0)

def reward_tracking_ang_vel(commands: jax.Array, x: Transform, xd: Motion, tracking_sigma) -> jax.Array:
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.clip(jp.exp(-ang_vel_error / (tracking_sigma + EPS)), -1000.0, 1000.0)

def reward_feet_air_time(air_time: jax.Array, first_contact: jax.Array, commands: jax.Array, minimum_airtime: float = 0.1) -> jax.Array:
    rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
    rew_air_time *= math.normalize(commands[:3])[1] > 0.05
    return jp.clip(rew_air_time, -1000.0, 1000.0)

def reward_abduction_angle(joint_angles: jax.Array, desired_abduction_angles: jax.Array = jp.zeros(4)):
    return jp.clip(jp.sum(jp.square(joint_angles[1::3] - desired_abduction_angles)), -1000.0, 1000.0)

def reward_stand_still(commands: jax.Array, joint_angles: jax.Array, default_pose: jax.Array, command_threshold: float) -> jax.Array:
    return jp.clip(jp.sum(jp.abs(joint_angles - default_pose)) * (math.normalize(commands[:3])[1] < command_threshold), -1000.0, 1000.0)

def reward_foot_slip(pipeline_state: base.State, contact_filt: jax.Array, feet_site_id: np.array, lower_leg_body_id: np.array) -> jax.Array:
    pos = pipeline_state.site_xpos[feet_site_id]
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    return jp.clip(jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))), -1000.0, 1000.0)

def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    return done & (step < step_threshold)

def reward_geom_collision(pipeline_state: base.State, geom_ids: np.array) -> jax.Array:
    contact = jp.array(0.0)
    for id in geom_ids:
        contact += jp.sum(((pipeline_state.contact.geom1 == id) | (pipeline_state.contact.geom2 == id)) * (pipeline_state.contact.dist < 0.0))
    return jp.clip(contact, -1000.0, 1000.0)

def reward_wheels_contact(contact_filt: jax.Array) -> jax.Array:
    return jp.clip(jp.sum(contact_filt), -1000.0, 1000.0)
'''
    with open(os.path.join(lib_path, "rewards.py"), "w") as f:
        f.write(rewards_code)
    print("Updated rewards.py")

    # 3. OVERWRITE ENVIRONMENT.PY
    # We need to read the existing file and inject the line, or overwrite a known version. 
    # Since environment.py is HUGE, we will use a smart replace.
    env_path = os.path.join(lib_path, "environment.py")
    with open(env_path, "r") as f:
        env_content = f.read()
    
    if '"ang_vel_z": rewards.reward_ang_vel_z(xd),' not in env_content:
        # Inject after ang_vel_xy
        target = '"ang_vel_xy": rewards.reward_ang_vel_xy(xd),'
        replacement = '"ang_vel_xy": rewards.reward_ang_vel_xy(xd),\n            "ang_vel_z": rewards.reward_ang_vel_z(xd),'
        env_content = env_content.replace(target, replacement)
        with open(env_path, "w") as f:
            f.write(env_content)
        print("Updated environment.py (injected ang_vel_z)")
    else:
        print("environment.py already up to date")

    # 4. OVERWRITE CONFIG.PY
    config_path = os.path.join(lib_path, "config.py")
    with open(config_path, "r") as f:
        conf_content = f.read()
    
    if 'ang_vel_z=0.0,' not in conf_content:
        target = 'ang_vel_xy=0.0,'
        replacement = 'ang_vel_xy=0.0,\n                        ang_vel_z=0.0,'
        conf_content = conf_content.replace(target, replacement)
        with open(config_path, "w") as f:
            f.write(conf_content)
        print("Updated config.py (injected ang_vel_z default)")
    else:
        print("config.py already up to date")

    print("\n--- PATCH COMPLETE ---")
    print("Please RESTART RUNTIME now if you haven't already, then skip the 'Clone/Install' cell.")
