#!/usr/bin/env bash


python gen_depth_point.py --loadckpt "./models/multi_s/14_0.061525317719469244.pt" \
    --outdir ../TankandTemples/advanced/ \
    --dataset tankstemple \
    --testlist ./lists/tankstemple/advanced.txt \
    --number_views_pred 7 \
    --iter 2 8 \
    --run_depth  \
    --run_fusion  \

python gen_depth_point.py --loadckpt "./models/multi_s/14_0.061525317719469244.pt" \
    --outdir ../TankandTemples/intermediate/ \
    --dataset tankstemple \
    --testlist ./lists/tankstemple/intermediate.txt \
    --number_views_pred 7 \
    --iter 2 8 \
    --run_depth  \
    --run_fusion  \