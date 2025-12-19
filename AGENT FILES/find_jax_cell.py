import json

notebook_path = r"c:\Users\tundt\Desktop\Robotics\Pupper\Pupper wheel\Pupper-wheels\wheelcolab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "jax[" in source or "jax" in source and "pip install" in source:
            print(f"Cell {i}:")
            print(source)
            print("-" * 20)
