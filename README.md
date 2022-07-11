# ADAC-traffic

## Offline reinforcement learning for traffic signal control
Paper will be uploaded here soon.

## Installation
1. Use the yml file provided in the folder to create a conda environment.
2. Install [sumo library](https://www.eclipse.org/sumo/) for traffic simulation.

## Data

Folder 'buffers' provides a small data set collected from cyclic traffic signal control policy.

To generate data sets with different sizes and behavioral policy, check the functionality provided in run-offline-rl.py program.

## Policy building and evaluation

Use the script eval-dac-policies.sh to try out model-based offline RL solutions using the data set provided in folder buffers.
