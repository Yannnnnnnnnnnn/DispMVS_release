# DispMVS_release
This is the official source code of the paper 'Rethinking Disparity: A Depth Range Free Multi-View Stereo Based on Disparity.'

---


## Dataset

Please follow the instruction from [IterMVS](https://github.com/FangjinhuaWang/IterMVS/blob/main/README.md).


---


## train

please check the 'train_blend_aug.sh' and 'train_dtu_aug.sh' and update the 'data_root' to your folder.


## eval

Please check the 'eval_dtu.sh' and 'eval_tanks.sh' to generate point clouds and update the 'outdir' to your folder.

As for evaluation for DTU, please run the Matlab code under 'evaluations/dtu'.
The results look like this:

| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.354    | 0.324     | 0.339        |


---


## TODO


more guide
release trained models


---


If you find this project useful for your research, please cite the following:
```
@article{yan2022rethinking,
  title={Rethinking Disparity: A Depth Range Free Multi-View Stereo Based on Disparity},
  author={Yan, Qingsong and Wang, Qiang and Zhao, Kaiyong and Li, Bo and Chu, Xiaowen and Deng, Fei},
  journal={arXiv preprint arXiv:2211.16905},
  year={2022}
}
```