import json
import os

NOTEBOOK_PATH = r"c:/Users/tundt/Desktop/Robotics/Pupper/Pupper wheel/Pupper-wheels/Simple colab/simple_wheelcolab.ipynb"

def modify_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    cells_modified = 0
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            modified_this_cell = False
            
            for line in source:
                # Modify Training Config
                if "training_config.lin_vel_x_range =" in line:
                    new_line = "training_config.lin_vel_x_range = [0.75, 0.75]  # min max [m/s]. FIXED for velocity tracking\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: lin_vel_x_range")
                    else:
                        new_source.append(line)
                elif "training_config.zero_command_probability =" in line:
                    new_line = "training_config.zero_command_probability = 0.0\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: zero_command_probability")
                    else:
                        new_source.append(line)
                
                # Modify Reward Config
                elif "reward_config.rewards.scales.tracking_ang_vel =" in line:
                    new_line = "reward_config.rewards.scales.tracking_ang_vel = 1.5\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: tracking_ang_vel")
                    else:
                        new_source.append(line)
                elif "reward_config.rewards.scales.abduction_angle =" in line:
                    new_line = "reward_config.rewards.scales.abduction_angle = 0.0\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: abduction_angle")
                    else:
                        new_source.append(line)
                elif "reward_config.rewards.scales.orientation =" in line:
                    new_line = "reward_config.rewards.scales.orientation = 0.0\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: orientation")
                    else:
                        new_source.append(line)
                elif "reward_config.rewards.scales.tracking_orientation =" in line:
                    new_line = "reward_config.rewards.scales.tracking_orientation = 0.0\n"
                    if new_line != line:
                        new_source.append(new_line)
                        modified_this_cell = True
                        print("Modified: tracking_orientation")
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            
            if modified_this_cell:
                cell['source'] = new_source
                cells_modified += 1

    if cells_modified > 0:
        try:
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Successfully modified {cells_modified} cells in {NOTEBOOK_PATH}")
        except Exception as e:
            print(f"Error writing notebook: {e}")
    else:
        print("No changes needed or targets not found.")

if __name__ == "__main__":
    modify_notebook()
