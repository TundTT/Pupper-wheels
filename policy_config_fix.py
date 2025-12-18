policy_config = config_dict.ConfigDict()

policy_config.use_imu = True # Whether to use IMU in policy. Default: True

policy_config.observation_history = 20  # number of stacked observations to give the policy

# --- FIXED ACTION SCALE ---
# legs (position control) need small scale (0.75)
# wheels (velocity control) need LARGE scale (20.0) to reach target speeds
leg_scale = 0.75
wheel_scale = 20.0
# Pattern is [Hip, Thigh, Wheel] repeated 4 times
policy_config.action_scale = jp.array([leg_scale, leg_scale, wheel_scale] * 4)

policy_config.hidden_layer_sizes = (256, 128, 128, 128) # default (256, 128, 128, 128)

# RTNeural supports relu, tanh, sigmoid (not great), softmax, elu, prelu
# Swish was really good in terms of training but not supported in RTNeural rn
policy_config.activation = "elu"
