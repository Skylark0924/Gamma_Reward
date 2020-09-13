# Gamma-Reward Algorithm for Multi-Agent Traffic Signal Control

<p align ="center">
  <img src="./README.assets/1565523437373.png">
</ p>

This is a code repository of the paper '[Learning Scalable Multi-Agent Coordination by
Spatial Differentiation for Traffic Signal Control](https://arxiv.org/abs/2002.11874)'

## Installation
Maybe you need to create a specialised conda environment and then run following codes:
```
git clone https://github.com/Skylark0924/Gamma-Reward-Perfect.git
cd Gamma-Reward-Perfect
conda install environment.yaml
pip install -r requirements.txt
```
**Note**: The package of `Cityflow` may need to be installed manually. Please see their website for details.

## Training
```
python ray_gamma_reward.py
```
Arguments are listed in ray_gamma_reward.py, you can change them in the script or by using the terminal.
