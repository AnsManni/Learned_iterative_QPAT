# Learned iterative model based QPAT solvers (+UNET)

This repository includes codes and datasets used to compute numerical examples for the paper: Anssi Manninen, Janek Grohl, Felix Lucka, and Andreas Hauptmann "Towards robust quantitative photoacoustic tomography via learned iterative methods," **(Submitted)**

## Authors
Anssi Manninen, Research Unit of Mathematical Sciences, University of Oulu, Oulu, Finland <br />
Janek Grohl, ENI-G, a Joint Initiative of the University Medical Center Göttingen and the Max Planck Institute for Multidisciplinary Sciences <br />
Felix Lucka, Centrum Wiskunde \& Informatica <br />
Andreas Hauptmann, Research Unit of Mathematical Sciences, University of Oulu, Oulu, Finland <br />

## Summary of the work


## Overview of the codes
The codes in this repository are used to train learned iterative model based solvers or (single step) U-Net to solve two different optical problems of QPAT.
The light propagation model is implemented using finite element approximation of **diffuse approximation** (DA) 

### Ideal problem
The ideal problem consists of a comprehensive amount of 1250 samples, with inclusions of randomly shaped and located ellipses. The used magnitude of optical values produce a highly diffusive region where the overall modeling error from using DA is negligible. This setup is ideal for comparing the convergence and reconstruction accuracy of the implemented methods. The photon fluence field of the samples was simulated using ValoMC software (https://inverselight.github.io/ValoMC/).



### Digital twin problem

### Implemented solvers 

**Learned iterative model-based solvers** <br />
Greedy / end-to-end training 
Gradient descent / Gauss-Newton / rank-1-update direction used as the information for the networks <br />

**U-Net** <br />
(single step) U-Net <br />

### Extendability of the codes ### 




## How to setup
To succesfully run the training scripts follow these steps:

**1.) Setup python environment** <br />
    - Either via Anaconda <br />
    - In terminal via 'conda create'

**2.) install following packages in Python enviroment:** <br />

'pip install ...':  <br />
    - matplotlib <br />
    - numpy <br />
    - scipy <br />
    - pickle <br />
    - pytorch (version X) <br />
    - tensorboardX (for visualiation) <br />
    - os <br />
    - time <br />

**3.) Choose either** <br />
Ideal problem (see README in Learned_QPAT_multi_illumination folder) <br />
or <br />
digital twin problem (see README in Learned_QPAT_multi_frequency folder) 




