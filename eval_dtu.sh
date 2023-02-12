#!/usr/bin/env bash

python gen_depth_point.py --loadckpt "./models/multi_ev/dtu_14_0.05775970327623216.pt" \
    --outdir "../dtu_eval" \
    --dataset dtu \
    --testlist ./lists/dtu/test.txt \
    --number_views_pred 5 \
    --iter 2 8 \
    --run_depth  \
    --run_fusion  \
    
