# --- COPY THIS BLOCK INTO THE 'Reward Configuration' CELL in wheelcolab.ipynb ---
# This replaces the entire existing reward_config setup.

reward_config = config_dict.ConfigDict()
reward_config.rewards = config_dict.ConfigDict()
reward_config.rewards.scales = config_dict.ConfigDict()

# 1. BOOSTED: Track linear velocity (The main goal)
# Increased from 1.5 to 4.0 to make forward movement the primary profit source
reward_config.rewards.scales.tracking_lin_vel = 4.0

# 2. Track the angular velocity along z-axis, i.e. yaw rate.
reward_config.rewards.scales.tracking_ang_vel = 1.5

# 3. REDUCED: Track the given body orientation
# Decreased from 5.0 to 1.0. This reduces the incentive to "turtle"/spin for stability.
reward_config.rewards.scales.tracking_orientation = 1.0

# Regularization Configs
reward_config.rewards.scales.lin_vel_z = -2.0
reward_config.rewards.scales.ang_vel_xy = -1.0

# 4. REDUCED: Penalize non-zero roll and pitch angles
# Decreased from -5.0 to -1.0 to be less punishing of the natural wobbles during rolling.
reward_config.rewards.scales.orientation = -1.0

reward_config.rewards.scales.torques = -0.0002
reward_config.rewards.scales.joint_acceleration = -0.001
reward_config.rewards.scales.mechanical_work = 0
reward_config.rewards.scales.action_rate = -0.05
reward_config.rewards.scales.feet_air_time = 0.0
reward_config.rewards.scales.wheels_contact = 1.0
reward_config.rewards.scales.stand_still = 0
reward_config.rewards.scales.stand_still_joint_velocity = 0
reward_config.rewards.scales.abduction_angle = -2.0
reward_config.rewards.scales.termination = -100.0
reward_config.rewards.scales.foot_slip = 0
reward_config.rewards.scales.knee_collision = -1.0
reward_config.rewards.scales.body_collision = -1.0
reward_config.rewards.tracking_sigma = 0.1

# --- COPY THIS LINE INTO THE 'Training Config' CELL in wheelcolab.ipynb ---
# Insert this near where other ranges (lin_vel_x_range) are defined.
# This restricts the robot to ONLY try straight lines for now, removing the option to spin.

training_config.ang_vel_yaw_range = [0.0, 0.0]  # Default was [-2.0, 2.0]
