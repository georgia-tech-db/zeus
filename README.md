# Zeus: Efficiently Localizing Actions in Videos using Reinforcement Learning


This repository contains the code for running Zeus inference experiments on the [BDD100k](https://www.bdd100k.com/) dataset.

The code requires:

Anaconda >= 4.10 \
Python >= 3.8 \
PyTorch >= 1.6.0

We assume that users have an existing installation of Anaconda. All the other dependencies can be installed by installing our conda environment:

```bash
conda env create -f environment.yml
conda activate zeus
```

This repository supports execution on both, CPU and GPU. However, we would recommend running the experiments on a GPU.




# RL Inference

The code for inference using trained RL models is in `src/rl-inference-engine`. The code follows the following module-format:

```bash

└── rl-inference-engine
    ├── dataset
    │   ├── bdd100k.py
    │   ├── custom_dataset.py
    │   └── custom_feat_extractor.py
    ├── models
    │   ├── dqn.py
    │   └── window_pp.py
    ├── rl-agent.py
    └── utils
        ├── constants.py
        ├── io.py
        └── misc.py
```

## Code walkthrough

1. `rl-agent.py` contains the code corresponding to the model inference process.
2. `dataset/` folder contains:
    * [bdd100k.py](src/rl-inference-engine/dataset/bdd100k.py) -- the `torch.dataset` utilities for the bdd100k dataset
    * [custom_dataset.py](src/rl-inference-engine/dataset/custom_dataset.py) -- code for navigating the video in an RL environment
    * [custom_feat_extractor.py](src/rl-inference-engine/dataset/custom_feat_extractor.py) -- code for extracting features on the fly using action recognition models.
3. `models/` folder contains the definitions for the Deep Learning model networks.
4. `utils/` folder contains utility files for inference.

## Model zoo

Please download the compressed zip file from [here](https://www.dropbox.com/s/qmrzkh60l0g3dlr/data.zip?dl=0).
The zip file contains the configuration files (models and metadata) for two action classes - `crossright` and `left`. We also provide the annotations (`data/datasets/bdd100k/labels.txt`) that we created for 5 action classes in BDD100k (for more details, please refer to the paper). Extract the zip file into the `data/` folder. After extraction, the `data` folder should follow the format:

```bash

└── data
    ├── datasets
    │   └── bdd100k
    └── models
        ├── action_reg_models
        │   ├── crossright
        │   └── left
        └── rl_models
            ├── crossright
            └── left
```

## Steps for setting up the dataset

1. To make sure that you can run the code, please download the BDD100k dataset into the `data/datasets/bdd100k` folder. Specifically, all the videos we use for our experiments can be found in the following splits of the BDD100k dataset:
```
1. bdd100k_videos_train_00.zip - http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_train_00.zip
2. bdd100k_videos_train_01.zip - http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_train_01.zip
3. bdd100k_videos_train_02.zip - http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_train_02.zip
```
2. You can find the videos that we use in the experiments along with the annotated temporal labels in the  file `data/datasets/bdd100k/labels.txt`.
3. Once downloaded, extract individual frames from the videos and place them in the folder `data/datasets/bdd100k/video_frames/`. The final folder should follow the format:

```bash
└── bdd100k
    ├── video_frames
    │   ├── 0000f77c-6257be58
    │   │   ├── frame000001.jpg
    │   │   ├── frame000002.jpg
    │   │   ├──     ...
    │   │   └── frame001216.jpg
    │   ├──   ...
    │   ├──   ...
    │   └── 05d34177-b978ae9f
    │      ├── frame000001.jpg
    │      ├── frame000002.jpg
    │      ├──     ...
    │      └── frame001206.jpg
    └── labels.txt

```

5. Run the inference script by using the command:
```bash
cd src/rl-inference-engine/
python rl-agent.py --dataset bdd100k --class-name 'crossright'
```

The results log will be generated in the `data/results` folder.
