# Uplift-Modeling

This is the Final Project for ML 2021 course in Skoltech. 

The repository contains reproducible source code for comparison of classical and state-of-the-art approaches to Uplift Modeling.

## Description

Uplift modeling is a new machine learning approach that allows you to evaluate the effect of a treatment and optimize the effectiveness of marketing campaigns, individual offers, or personalized medicine. The uplift modeling is used to determine the user groups that will be most affected by the campaign treatment and the groups that do not need to be affected.

In this project, we review standard approaches to uplift modeling, as well as completely new and innovative ones. We compare existing libraries in this field and reproduce the results of two state-of-the-art articles and suggest improvements. We evaluate the performance of the reviewed models using both synthetic and real data.

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

TBD