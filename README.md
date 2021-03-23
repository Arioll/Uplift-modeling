# Uplift-Modeling

This is the Final Project for ML 2021 course in Skoltech. 

The repository contains reproducible source code for comparison of classical and state-of-the-art approaches to Uplift Modeling.

## Description

Uplift modeling is a new machine learning approach that allow to evaluate the effect of a treatment and optimize the effectiveness of marketing campaigns, individual offers, or personalized medicine. The uplift modeling is used to determine the user groups that will be most affected by the campaign treatment and the groups that do not need to be affected.

In this project, we review standard approaches to uplift modeling, as well as completely new and innovative ones. We compare existing libraries in this field, reproduce the results of two state-of-the-art articles and suggest improvements. We evaluate the performance of the reviewed models using both synthetic and real data.

## Prerequisites

Main prerequisites are:

- [pytorch](http://pytorch.org/)
- [causaml](https://github.com/uber/causalml)
- [econml](https://github.com/microsoft/EconML)
- [scikit-uplift](https://github.com/maks-sh/scikit-uplift)

## How to use

1. Clone GitHub repository
2. Run Jupyter notebooks

## Notebooks

- `learners_by_models.ipynb` -- Comparison of Meta-Learners approach with different base models on various data sets using uplift at k metric
- `learners_by_libraries.ipynb` -- Comparison of Python libraries for Meta-Learners approach
on various data sets using uplift at k metric
- `forest_by_functions.ipynb` -- Comparison of Evaluation Functions for Uplift Random
Forest approach on various data sets using uplift at k metric
- `SMITE.ipynb` -- Replication of [SMITE](https://arxiv.org/abs/2011.00041) architecture and comparison with Meta-Learners approach on various data sets using uplift at k metric
- `RLift.ipynb` -- Replication of ["Reinforcement learning for uplift modeling"](https://arxiv.org/abs/1811.10158) article

## Results

- We compared Meta-Learners implemented in different libraries with each other and Meta-Learners with tree-based approach. From the comparison, we can conclude that the CausalML, Sklift and EconML libraries have equivalent implementations of Meta-Learners. 
- Among the Meta-Learners and trees, it is difficult to determine the best model. The choice of the algorithm depends on the data. 
- We also found that a fully connected network as a base model for Meta-Learners does not provide a significant benefit compared to boosting. 
- However, we reproduced a more complex architecture of the neural network presented in  [SMITE](https://arxiv.org/abs/2011.00041) which shows significantly better results. 
- We also reproduced a promising ["Reinforcement learning for uplift modeling"](https://arxiv.org/abs/1811.10158) approach and obtained a SN-UMG of 0.83 on a Synthetic dataset. 

As a future work, we plan to evaluate the RL approach more objectively and make the SMITE architecture more generic for different data.
