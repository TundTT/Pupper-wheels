import json

notebook_path = r"c:\Users\tundt\Desktop\Robotics\Pupper\Pupper wheel\Pupper-wheels\wheelcolab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        changed = False
        for line in cell['source']:
            if '!pip install jax==0.6.0 jaxlib==0.6.0' in line:
                # Replace with stable 0.4.30
                new_source.append('!pip install "jax[cuda12_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n')
                changed = True
            else:
                new_source.append(line)
        if changed:
            cell['source'] = new_source
            print("Updated JAX installation cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
