# O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning

![Overview](/images/teaser.png)

**Object-object Interaction Affordance Learning.** For a given object-object interaction task (e.g., fitting), our method takes as inputs a 3D acting object point cloud (the bucket) and a partial 3D scan of the scene/object (the cabinet), and outputs an affordance prediction heatmap that estimates the likelihood of the acting object successfully accomplishing the task at every pixel. At the test time, one may easily sample a position from the heatmap to perform the action.

## Introduction

Contrary to the vast literature in modeling, perceiving, and understanding agent-object (e.g., human-object, hand-object, robot-object) interaction in computer vision and robotics, very few past works have studied the task of object-object interaction, which also plays an important role in robotic manipulation and planning tasks. There is a rich space of object-object interaction scenarios in our daily life, such as placing an object on a messy tabletop, fitting an object inside a drawer, pushing an object using a tool, etc. In this paper, we propose a unified affordance learning framework to learn object-object interaction for various tasks. By constructing four object-object interaction task environments using physical simulation (SAPIEN) and thousands of ShapeNet models with rich geometric diversity, we are able to conduct large-scale object-object affordance learning without the need for human annotations or demonstrations. At the core of technical contribution, we propose an object-kernel point convolution network to reason about detailed interaction between two objects. Experiments on large-scale synthetic data and real-world data prove the effectiveness of the proposed approach.

## About the paper

O2O-Afford is accepted to CoRL 2021!

Our team: 
[Kaichun Mo](https://cs.stanford.edu/~kaichun),
[Yuzhe Qin](https://yzqin.github.io/),
[Fanbo Xiang](https://www.fbxiang.com/),
[Hao Su](http://ai.ucsd.edu/~haosu/),
[Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/)
from 
Stanford University and UC San Diego.

ArXiv Version: https://arxiv.org/abs/2106.15087

Project Page: https://cs.stanford.edu/~kaichun/o2oafford/

## Citations

  @inProceedings{mo2021o2oafford,
      title={{O2O-Afford}: Annotation-Free Large-Scale Object-Object Affordance Learning},
      author={Mo, Kaichun and Qin, Yuzhe and Xiang, Fanbo and Su, Hao and Guibas, Leonidas},
      year={2021},
      booktitle={Conference on Robot Learning (CoRL)}
  }

## Code Dependencies

This code is a Python-codebase and use PyTorch framework for deep learning. We use Pytorch 1.7.1 and TorchVision 0.8.2. The codebase has been tested on ubuntu 18.04, Python 3.8, Cuda 10.2.

Other dependencies (install via pip) are: SAPIEN 0.8, NumPy, H5py, matplotlib, pillow, progressbar2, pyyaml, six, scikit-learn, scipy, sklearn, tensorboard, trimesh, opencv-python.

It also uses external python libraries such as Chamfer-distance, PointNet2, SAPIEN that could be installed via
  * https://github.com/chrdiller/pyTorchChamferDistance
  * https://github.com/erikwijmans/Pointnet2_PyTorch
    (after installation, you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
  * https://github.com/haosulab/SAPIEN-Release

This code also depends on Blender v2.79b and Thea (https://github.com/sidch/Thea) for result visualization.


## Statement

For this code submission, due to the 100MB size limit of the supplementary zip file, we only attach a small sample of data if reviewers want to run the code to see our system in action.
This amount of data is sufficient to show our data generation process, visualize the four task environments we use in the paper, and test over our pre-trained models.
However, this amount of data is definitely not enough for a successful training.
We will release the full data once the paper is accepted for the final code and data release.


## Code Structure

The main directory includes:
1) `data` stores the SAPIEN shape data and the collected interaction trial data (Here, for this code submission, due to the 100MB size limit of the supplementary zip file, we only attach a small sample of data; but we will release the full data once the paper is accepted for the final code and data release).
2) `exps` contains all code implementation, as well as training and test scripts, logs, results, etc.
3) `stats` contains necessary setting files specifying the data statistics to use to run experiments over the released small set of data.


### To visualize the task environment and try to collect one data

We release the code for the four environments: placement, fitting, pushing, and stacking.
For example, for placement, we release the code in folder `exps/env_placement/`, where `env.py` defines the environment, `collect_data.py` defines the interaction trials, `replay_data.py` presents a code that can replay a pre-collected data, and `stats` store the shape statistics used for this task.

To collect one data, go to the `exps` directory and run

    python env_placement/collect_data.py 40147 StorageFurniture

Change to other environments for other tasks.
This script will store a collected interaction data record at `results/env_placement/40147_StorageFurniture_0_0`.

After generating the data you can run the followig to replay this data

    python env_placement/replay_data.py results/env_placement/40147_StorageFurniture_0_0/

During collecting or replaying the data, an OpenGL window will pop out to visualze the task environment.


### To batch-collect data for training

Before training our O2O-Afford Perception network, we need to collect a large-scale interaction data from the SAPIEN environment.
We release a code `exps/gen_offline_data.py` to batch-generate the large-scale data.

Please go to the `exps` directory and run the following to generate data over the small set of data for preview.

    bash scripts/gen_data-placement.sh

Change to other environments for other tasks.

This code will call the script `gen_offline_data.py` which will spawn 8 CPUs to parallize the data generation, each of which calls the file `env_placement/collect_data.py` to generate one interaction data.

Running this script will generate three data folders `data/offlinedata-placement-train_cat_train_shape`, `data/offlinedata-placement-train_cat_test_shape` and `offlinedata-placement-test_cat`, each of which contains several subfolders of data interaction and a `data_tuple_list.txt` listing all valid interaction data.


### To train the network

Using the batch-generated interaction data, run the following to train the network

    bash scripts/train_placement.sh

Change to other environment names to train for other tasks.

The training will generate a log directory under `exps/logs/exp-placement`.

Notice that with such small amount of data, the training code will not generate a good performance. This code is just for reviewing purpose.


### To test the network using our pretrained model

We provide our pre-trained model for you to test over sample data. 
The pre-trained record is under `exps/logs/exp-placement-pretrained`.

You can run 

    bash scripts/eval_placement.sh

to test over some test data we provided in `./data/smalldata-placement`.

It will creates a folder `test-whole-model_epoch_12-succ_thres_50-train_cat_test_data` that includes a data visualization webpage.
Go to the following link to visualize in your browser.

    exps/logs/exp-placement-pretrained/test-whole-model_epoch_12-succ_thres_50-train_cat_test_data/visu/htmls/index.html

If you fail to run the code, we have also provided our generated result visualization page (over the entire full dataset) under

    exps/logs/exp-placement-pretrained/results/htmls/index.html


## External Libraries

This code uses the following external libraries (all are free to use for academic purpose):
   * https://github.com/sidch/Thea
   * https://github.com/erikwijmans/Pointnet2_PyTorch
   * https://github.com/haosulab/SAPIEN-Release
   * https://github.com/KieranWynn/pyquaternion

We use the data from SAPIEN and PartNet, which are both cited in the main paper.


## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.


## License

MIT Licence


## Updates

* [Nov 8, 2021] Preliminary vesion of Data and Code released.
