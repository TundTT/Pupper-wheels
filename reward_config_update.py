# --- Updated Reward Config for Smooth Rolling ---
# Copy and paste this into the Reward Configuration cell in your notebook.

reward_config = config_dict.ConfigDict()
reward_config.rewards = config_dict.ConfigDict()
reward_config.rewards.scales = config_dict.ConfigDict()

# Track linear velocity
reward_config.rewards.scales.tracking_lin_vel = 1.5

# Track the angular velocity along z-axis, i.e. yaw rate.
reward_config.rewards.scales.tracking_ang_vel = 1.5

# Track the given body orientation (desired world z axis in body frame)
# FIX: Changed from -5.0 to 5.0 to REWARD good orientation instead of penalizing it
reward_config.rewards.scales.tracking_orientation = 5.0

# Below are regularization terms, we roughly divide the
# terms to base state regularizations, joint
# regularizations, and other behavior regularizations.
# Penalize the base velocity in z direction, L2 penalty.
reward_config.rewards.scales.lin_vel_z = -2.0

# Penalize the base roll and pitch rate. L2 penalty.
# INCREASED: From -0.05 to -1.0 to dampen wobbling/weird dynamics
reward_config.rewards.scales.ang_vel_xy = -1.0

# Penalize non-zero roll and pitch angles. L2 penalty.
reward_config.rewards.scales.orientation = -5.0

# L2 regularization of joint torques, sum(|tau|^2).
reward_config.rewards.scales.torques = -0.0002

# L2 regularization of joint accelerations sum(|qdd|^2)
# INCREASED: From -5e-6 to -0.001 to suppress high-freq leg vibrations
reward_config.rewards.scales.joint_acceleration = -0.001

# L1 regularization of mechanical work, |v * tau|.
reward_config.rewards.scales.mechanical_work = 0

# Penalize the change in the action and encourage smooth
# actions. L1 regularization |action - last_action|^2
# INCREASED: From -0.01 to -0.05 for smoother control
reward_config.rewards.scales.action_rate = -0.05

# Encourage long swing steps. However, it does not
# encourage high clearances.
# FIX: Set to 0.0 because we don't want to encourage "air time" for wheels
reward_config.rewards.scales.feet_air_time = 0.0

# NEW: Encourage wheels to stay on the ground
reward_config.rewards.scales.wheels_contact = 1.0

# Encourage joints at default position at zero command, L1 regularization
# |q - q_default|.
reward_config.rewards.scales.stand_still = 0

# Encourage zero joint velocity at zero command, L1 regularization
# |q_dot|.
# Activates when norm(command) < stand_still_command_threshold
# Commands below this threshold are sampled with probability zero_command_probability
reward_config.rewards.scales.stand_still_joint_velocity = 0

# Encourage zero abduction angle so legs don't spread so far out
# L2 loss on ||abduction_motors - desired||^2
# INCREASED: From -0.1 to -2.0 to prevent legs from splaying out (spinning cause)
reward_config.rewards.scales.abduction_angle = -2.0

# Early termination penalty.
reward_config.rewards.scales.termination = -100.0

# Penalizing foot slipping on the ground.
reward_config.rewards.scales.foot_slip = 0

# Penalize knees hitting the ground
reward_config.rewards.scales.knee_collision = -1.0

# Penalize body hitting ground
reward_config.rewards.scales.body_collision = -1.0

# Tracking reward = exp(-error^2/sigma).
# TIGHTENED: From 0.25 to 0.1 to force stricter tracking
reward_config.rewards.tracking_sigma = 0.1
