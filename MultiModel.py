import os
import numpy as np


counter = 1
for gamma in [0.95, 0.97, 0.99, 1.00]:
    for fc1_units in [100, 200, 400]:
        for fc2_units in [100, 200, 400]:
            for LR_ACTOR in reversed([1e-4, 1e-5]):
                for LR_CRITIC in reversed([1e-4, 1e-5]):
                    os.system(f'python ./training.py \
                        --n_episodes 2000 \
                        --max_t 2000 \
                        --model_num {counter} \
                        --GAMMA {gamma} \
                        --LR_ACTOR {LR_ACTOR:.4f} \
                        --LR_CRITIC {LR_CRITIC:.4f} \
                        --fc1_units {fc1_units} \
                        --fc2_units {fc2_units} \
                             ')

                    counter += 1  
