# Experiments

## Before Start

To train the models, please first go to the `../data` folder and download the simulation assets/small-data/others.

Please follow the instruction below to generate large-scale data for training models from scratch.

To test over the pretrained models, please also go to the `./logs` folder and download the pretrained checkpoints.
We also provided a small subset of data under `./data/smalldata-placement`for you to test the pre-trained models.

Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSd7_Nqti_La1ROYjpjL8D-lkluIlswu2GJlnv_j3KyCcnMoYw/viewform?usp=sf_link) to download all resources.

## Dependencies

This code is a Python-codebase and use PyTorch framework for deep learning. We use Pytorch 1.7.1 and TorchVision 0.8.2. The codebase has been tested on ubuntu 18.04, Python 3.8, Cuda 10.2.

First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is still being actively developed and updated.

This codebase also depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    # [IMPORTANT] Also, you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
    pip install -r requirements.txt
    pip install -e .

(Optionally) For visualization, please install blender v2.79 (sorry the codebase does not work for the latest version of Blender) and put the executable in your environment path.
Also, the prediction result can be visualized using MeshLab or the *RenderShape* tool in [Thea](https://github.com/sidch/thea).

Other dependencies (install via pip) are: NumPy, H5py, matplotlib, pillow, progressbar2, pyyaml, six, scikit-learn, scipy, sklearn, tensorboard, trimesh, opencv-python.


## Code Structure

The main directory includes:
1) `data` stores the SAPIEN shape data and the collected interaction data.
2) `exps` contains all code implementation, as well as training and test scripts, logs (containing pretrained checkpoints and some result visualization), results, etc.
3) `stats` contains necessary setting files specifying the data statistics to use to run the experiments.


### To visualize the task environment and try to collect one data

We release the code for the four environments: placement, fitting, pushing, and stacking.
For example, for placement, we release the code in folder `exps/env_placement/`, where `env.py` defines the environment, `collect_data.py` defines the interaction trials, `replay_data.py` presents a code that can replay a pre-collected data, and `stats` store the shape statistics used for this task.

To collect one data, go to the `exps` directory and run

    python env_placement/collect_data.py 40147 StorageFurniture

Change to other environments for other tasks.
This script will store a collected interaction data record at `results/env_placement/40147_StorageFurniture_0_0`.

After generating the data you can run the following to replay this data

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

