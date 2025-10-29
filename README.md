# Learned iterative model based QPAT solvers (+UNET)

This repository includes codes and datasets used to compute numerical examples for the paper: Anssi Manninen, Janek Grohl, Felix Lucka, and Andreas Hauptmann "Towards robust quantitative photoacoustic tomography via
learned iterative methods," Journal of...


The codes in this repository are used to train learned iterative model based solvers to solve two different optical problems of QPAT.

The principal options for the training are: <br />
Greedy / end-to-end training / (single step) U-Net <br />
Gradient descent / Gauss-Newton / rank-1-update direction used as the information for the networks

To succesfully run the training scripts follow these steps:

1.) Setup python environment <br />
    - Either via Anaconda <br />
    - In terminal via 'conda create'

2.) install following packages in Python enviroment: <br />

'pip install ...':  <br />
    - matplotlib <br />
    - numpy <br />
    - scipy <br />
    - pickle <br />
    - pytorch (version X) <br />
    - tensorboardX (for visualiation) <br />
    - os <br />
    - time <br />

3.) <br />
Choose either the ideal problem (see README in Learned_QPAT_multi_illumination folder) <br />
or <br />
digital twin problem (see README in Learned_QPAT_multi_frequency folder) 




