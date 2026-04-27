import json, os

with open('notebooks/statecraft_train.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_mode = False
        for line in source:
            if 'TRAIN_MODE = "grpo"' in line:
                line = '    "TRAIN_MODE = \\"grpo\\"   # one of: grpo, standard, curriculum\\n",\n'
            elif 'elif TRAIN_MODE == "ppo":' in line:
                skip_mode = True
            elif 'elif TRAIN_MODE == "standard":' in line:
                skip_mode = False
            elif 'raise ValueError("TRAIN_MODE must be one of: grpo, ppo, standard, curriculum")' in line:
                line = '    "    raise ValueError(\\"TRAIN_MODE must be one of: grpo, standard, curriculum\\")\\n",\n'
            
            if not skip_mode:
                new_source.append(line)

        if any('OPTIONAL GENERALIZATION EVAL' in L for L in source):
            new_source = [
                '    "# ===== 4) GENERALIZATION EVAL =====\\n",\n',
                '    "from eval.generalization import run_generalization_test\\n",\n',
                '    "\\n",\n',
                '    "gen = run_generalization_test()\\n",\n',
                '    "with open(os.path.join(RUN_DIR, \\"generalization_results.json\\"), \\"w\\", encoding=\\"utf-8\\") as f:\\n",\n',
                '    "    json.dump(gen, f, indent=2)\\n",\n',
                '    "print(\\"Saved generalization_results.json\\")\\n"\n'
            ]

        cell['source'] = new_source

with open('notebooks/statecraft_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

with open('notebooks/statecraft_train.txt', 'w') as f:
    f.write('See colab_pipeline.py for the full pipeline. This text file is deprecated.')
