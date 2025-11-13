import mujoco
from mujoco import viewer
import os
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script's directory with the XML file name to get the full path
xml_path = os.path.join(script_dir, "Pupper_wheel_site_velocity_track.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()