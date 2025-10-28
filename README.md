# Learned_iterative_QPAT

This repository includes codes and datasets used to compute numerical examples for the paper: Anssi Manninen, Janek Grohl, Felix Lucka, and Andreas Hauptmann "Towards robust quantitative photoacoustic tomography via
learned iterative methods," Journal of...


The codes in this repository are used to train learned iterative model based solvers to solve two different optical problems of QPAT.

The principal options for the training are:

Greedy / end-to-end training 

Gradient descent / Gauss-Newton / rank-1-update direction used as the information for the networks

To succesfully run the training scripts install following packages in Python enviroment:

    'pip install ...':
    - matplotlib
    - numpy 
    - scipy
    - pickle
    - pytorch (version X)
    - tensorboardX (for visualiation)
    - os
    - time


The ideal (multi-illumination) problem and link to related data set can be found from "Learned_QPAT_multi_illumination" folder

The digital twin problem and link to related data set can be fround from ... 






Digital twin...:




