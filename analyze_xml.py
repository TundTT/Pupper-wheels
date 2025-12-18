
import mujoco
import numpy as np
import os

model_path = r"c:/Users/tundt/Desktop/Robotics/Pupper/Pupper wheel/Pupper-wheels/description/mujoco_xml/Pupper_wheel_site_velocity_track.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Forward kinematics to get initial state
    mujoco.mj_forward(model, data)
    
    print(f"Model loaded: {model_path}")
    
    wheel_joints = ["wheel_front_r", "wheel_front_l", "wheel_back_r", "wheel_back_l"]
    
    for j_name in wheel_joints:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id == -1:
            print(f"Joint {j_name} not found!")
            continue
            
        # Get joint axis in global frame
        # xaxis is the joint axis in local body frame? No, MjModel stores it in local.
        # We need to transform it by the body orientation.
        body_id = model.jnt_bodyid[j_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        # Global axis = Body_Rotation_Matrix * Local_Axis
        # Data.xmat gives body orientation in global frame (3x3 flattened)
        xmat = data.xmat[body_id].reshape(3, 3)
        local_axis = model.jnt_axis[j_id]
        global_axis = xmat @ local_axis
        
        print(f"Joint: {j_name} (Body: {body_name})")
        print(f"  Local Axis: {local_axis}")
        print(f"  Global Axis: {global_axis}")
        
        # Check orthogonality with Forward (X) and Up (Z)
        # Ideal rolling axis for X-movement is Y-axis (0, 1, 0)
        # So we expect Global Axis to be roughly (0, 1, 0) or (0, -1, 0)
        print(f"  Alignment with Y-axis: {np.dot(global_axis, np.array([0, 1, 0])):.3f}")
        
except Exception as e:
    print(f"Error: {e}")
