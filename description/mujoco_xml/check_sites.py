import mujoco
import os
import sys

def check_xml(xml_filename, expected_sites):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, xml_filename)
    print(f"Checking {xml_path}...")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"Failed to load XML: {e}")
        return

    all_found = True
    for name in expected_sites:
        id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name)
        if id == -1:
            print(f"  Site '{name}' NOT FOUND!")
            all_found = False
        else:
            print(f"  Site '{name}' found (id={id})")
    
    if all_found:
        print("All sites found successfully.")
    else:
        print("Some sites were missing.")

if __name__ == "__main__":
    print("--- Checking Pupper_wheel_site_velocity_track.xml ---")
    # New names expected here
    new_sites = [
        "wheel_front_r_site",
        "wheel_front_l_site",
        "wheel_back_r_site",
        "wheel_back_l_site"
    ]
    check_xml("Pupper_wheel_site_velocity_track.xml", new_sites)

    print("\n--- Checking Pupper_wheel_site.xml ---")
    # Old names expected here (unchanged)
    old_sites = [
        "wheel_front_r_foot_site",
        "wheel_front_l_foot_site",
        "wheel_back_r_foot_site",
        "wheel_back_l_foot_site"
    ]
    check_xml("Pupper_wheel_site.xml", old_sites)
