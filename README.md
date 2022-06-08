## How Far I'll Go:</br>Offline Goal-Conditioned Reinforcement Learning via </br>f-Advantage Regression

#### [[Project Page]](https://jasonma2016.github.io/GoFAR/) [[Paper]](https://arxiv.org/abs/2206.03023)

[Jason Yecheng Ma](https://www.seas.upenn.edu/~jasonyma/)<sup>1</sup>, [Jason Yan](https://www.linkedin.com/in/jasyan/)<sup>1</sup>, [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/)<sup>1</sup>, [Osbert Bastani](https://obastani.github.io/)<sup>1</sup>

<sup>1</sup>University of Pennsylvania

This is a PyTorch implementation of our paper [How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via F-Advantage Regression](); this code can be used to reproduce Section 5.1 and 5.2 of the paper. 

Here is a teaser video comparing GoFAR against state-of-art offline GCRL algorithms on a real robot!
<img src="media/dclawturn_policies.gif" width="550">

## SetUp
### Requirements
- MuJoCo=2.0.0

### Setup Instructions
1. Create conda environment and activate it:
     ```
     conda env create -f environment.yml
     conda activate gofar
     pip install --upgrade numpy
     pip install torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0 gym==0.17.3
2. (Optionally) install the [Robel](https://github.com/google-research/robel) environment for the D'Claw experiment.
3. Download the offline dataset [here](https://drive.google.com/file/d/1niq6bK262segc7qZh8m5RRaFNygEXoBR/view) and place ```/offline_data``` in the project root directory.

## Experiments
We provide commands for reproducing the main GCRL results (Table 1), the ablations (Figure 3), and the stochastic offline GCRL experiment (Figure 4). 

1. The main results (Table 1) can be reproduced by the following command:
```
mpirun -np 1 python train.py --env $ENV --method $METHOD
```
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--env $ENV``  | offline GCRL tasks: ```FetchReach, FetchPush, FetchPick, FetchSlide, HandReach, DClawTurn```|
| ``--method $METHOD``  | offline GCRL algorithms: ```gofar, gcsl, wgcsl, actionablemodel, ddpg```|

2. To run the ablations (Figure 3), we can adjust some relevant command arguments. For example, to disable HER, we can do
```
mpirun -np 1 python train.py --env $ENV --method $METHOD --relabel False
```
Note that ```gofar``` defaults to not using HER, so this command is only relevant to the baselines. Relevant flags are listed here:
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--relabel``  | whether hindsight experience replay is enabled: ``True``, ``False  ``|
| ``--relabel_percent``  | The fraction of minibatch transitions that has relabeled goals: ``0.0, 0.2, 0.5, 1.0``; these are the hyperparameters attempted in the paper, you may try other fractions too.|
| ``--f``  | Choices of f-divergence for GoFAR: ``kl, chi``.
| ``--reward_type``  | Choices of reward function for GoFAR: ``disc, binary``.

3. The following command will run the stochastic environment experiment (Figure 4):
```
mpirun -np 1 python train.py --env FetchReach --method $METHOD --noise True --noise-eps $NOISE_EPS
```
where ```$NOISE_EPS``` can be chosen from ```0.5, 1.0, 1.5```.

## Acknowledgement:
We borrowed some code from the following repositories:
- [Pytorch DDPG-HER Implementation](https://github.com/TianhongDai/hindsight-experience-replay)
- [AWGCSL](https://github.com/YangRui2015/AWGCSL)
- [GCSL](https://github.com/dibyaghosh/gcsl)
- [Openai Baselines](https://github.com/openai/baselines)
