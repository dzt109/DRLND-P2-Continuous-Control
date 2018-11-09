import os
import numpy as np
import glob
import json

best_models = [130, 94, 60, 96, 132]

for model in best_models:
    params = glob.glob(f'results-reacher/model-{model}/*.json')[0]
    with open(params) as f:
        data = json.load(f)
    
        for key, value in data.items():
            exec(f'{key} = {value}')
    
        for idx, seed in enumerate([10,20,30,40,50]):
            model_num = f'{model}{seed}'
            if os.path.isfile(f'results/model-{model_num}/scores.png'):
                print (f'model-{model_num} is already done')
                continue
            
            if idx == 4:
                os.system(f'python3 ./training.py \
                         --n_episodes 400 \
                         --max_t 2000 \
                         --model_num {model_num} \
                         --GAMMA {GAMMA} \
                         --LR_ACTOR {LR_ACTOR:.4f} \
                         --LR_CRITIC {LR_CRITIC:.4f} \
                         --fc1_units {fc1_units} \
                         --fc2_units {fc2_units} \
                         --seed {seed}')
            else:
                os.system(f'nohup python3 ./training.py \
                         --n_episodes 400 \
                         --max_t 2000 \
                         --model_num {model_num} \
                         --GAMMA {GAMMA} \
                         --LR_ACTOR {LR_ACTOR:.4f} \
                         --LR_CRITIC {LR_CRITIC:.4f} \
                         --fc1_units {fc1_units} \
                         --fc2_units {fc2_units} \
                         --seed {seed} &')