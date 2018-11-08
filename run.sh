#!/bin/sh
python -u ./LENA.py --dim 200 --neg_weight 0.5 --L 5 --H 90 --batch 200 --data ./data/FB15K-237/ --eval_per 5 --save_per 5 --worker 10 --eval_batch 500 --max_iter 30 --generator 10