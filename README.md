# Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds

This repository contains the implementation of our CVPR2020 paper Multi Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds(MPRM)([paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Path_Region_Mining_for_Weakly_Supervised_3D_Semantic_Segmentation_on_CVPR_2020_paper.pdf))



## Installation
This project is developed based on Kernel Point Convolution [KPConv](https://arxiv.org/abs/1904.08889) ([repo](https://github.com/HuguesTHOMAS/KPConv)).
You can follow the original KPConv [installation guide](https://github.com/HuguesTHOMAS/KPConv/blob/master/INSTALL.md) for ubuntu 16.04 and 18.04.

If you want to use [dense-crf](https://arxiv.org/abs/1210.5644) for post-processing, install pydense-crf following this [repo](https://github.com/lucasb-eyer/pydensecrf)

## Training
1. Download the Scannet dataset through the [official webcite](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation).

2. Modify the dataset path in `datasets/Scannet_subcloud.py` And start training:
```
python training_mprm.py
```
You can also specify the model saving path and change parameters in `training_mprm.py`.

3. You can plot the training details by speficy the saved model path and run:
```
python plot_convergence_mprm.py
```

4. Generate the pseudo label by running:
```
python generate_pseudo_label.py
```

5. You can choose to post-process the pseudo-label by running:
```
python crf_postprocessing.py
```

6. Finally, use the pseudo label to train a segmentation network by running:
```
python training_segmentation.py
```


## Citing this work
If you find this work useful, please cite:
[Multi Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Path_Region_Mining_for_Weakly_Supervised_3D_Semantic_Segmentation_on_CVPR_2020_paper.pdf)
```
@inproceedings{wei2020multi,
  title={Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds},
  author={Wei, Jiacheng and Lin, Guosheng and Yap, Kim-Hui and Hung, Tzu-Yi and Xie, Lihua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4384--4393},
  year={2020}
}
```

## Acknowledgement
This project is developed based on [KPConv](https://arxiv.org/abs/1904.08889) ([repo](https://github.com/HuguesTHOMAS/KPConv)).
We also thanks  [dense-crf](https://arxiv.org/abs/1210.5644)[repo](https://github.com/lucasb-eyer/pydensecrf) and [nanoflann](https://github.com/jlblancoc/nanoflann).

## License
Our code is released under MIT License (see LICENSE file for details).

