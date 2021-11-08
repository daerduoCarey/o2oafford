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


## Code

Please go to `exps` folder and refer to the README there.

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.


## License

MIT Licence


## Updates

* [Nov 8, 2021] Preliminary vesion of Data and Code released.
