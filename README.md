# Learned iterative model based QPAT solvers (+UNET)

This repository includes codes and datasets used to compute numerical examples for the paper: *"Towards robust quantitative photoacoustic tomography via learned iterative methods"* **(Submitted)**

## Authors
Anssi Manninen<sup>1</sup>  <br />
Janek Grohl<sup>2</sup> <br />
Felix Lucka<sup>3</sup>  <br />
Andreas Hauptmann<sup>1,4</sup>  <br />

## Affiliations 
1. Research Unit of Mathematical Sciences, University of Oulu, Oulu, Finland <br />
2. ENI-G, a Joint Initiative of the University Medical Center Göttingen and the Max Planck Institute for Multidisciplinary Sciences, Göttingen, Germany <br />
3. Centrum Wiskunde \& Informatica, Amsterdam, The Netherlands <br />
4. Department of Computer Science, University College London, London, United Kingdom <br />

## Summary of the work
Photoacoustic tomography (PAT) is a medical imaging modality that can provide high-resolution tissue images based on the optical absorption. Classical reconstruction methods for quantifying the absorption coefficients rely on sufficient prior information to overcome noisy and imperfect measurements. As these methods utilize computationally expensive forward models, the computation becomes slow, limiting their potential for time-critical applications. As an alternative approach, deep learning-based reconstruction methods have been established for faster and more accurate reconstructions. However, most of these methods rely on having a large amount of training data, which is not the case in practice. In this work, we adopted the model-based learned iterative approach for the use in Quantitative PAT (QPAT), in which additional information from the model is iteratively provided to the updating networks, allowing better generalizability with scarce training data. 

The codes provided in this repository implement iterative learned model-based updates based on **gradient descent**, **Gauss-Newton**, and **rank-1-update** methods that can be used to solve two (2D) optical problems of QPAT, the **ideal problem** and the **digital twin problem** described below. The learning task can be selected to be formulated as **greedy** or **end-to-end**. For comparison learned single-step reconstruction method based on **U-Net** architecture is implemented. To solve the optical problem, the light propagation is modeled by using finite element approximation of **diffuse approximation** (DA). See [Implemented solvers](#Implemented-solvers) for details of the implementations.


### Ideal problem
The ideal problem consists of a comprehensive amount of 1250 samples, with inclusions of randomly shaped and located ellipses. The used magnitude of optical values produce a highly diffusive region where the overall modeling error from using DA is negligible. This setup is ideal for comparing the convergence and reconstruction accuracy of the implemented methods. The photon fluence field of the samples was simulated using ValoMC software (https://inverselight.github.io/ValoMC/).

### Digital twin problem
The digital twin problem is much closer to experimental conditions compared to the ideal problem. The problem now has limited training data, samples with varying locations of the tubes (region of interest), a large range of possible optical values, and a large modeling error. The digital twin problem re-used data from physical tissue-mimicking phantoms with piecewise-constant material distributions cite{}


## Overview of the codes
The codes in this repository are used to train learned iterative model based solvers or (single step) U-Net to solve two different optical problems of QPAT.
The light propagation model is implemented using finite element approximation of **diffuse approximation** (DA) 



### Implemented solvers 

**Learned iterative model-based solvers** <br />
Greedy / end-to-end training 
Gradient descent / Gauss-Newton / rank-1-update direction used as the information for the networks <br />

**U-Net** <br />
(single step) U-Net <br />

### Extendability of the codes 
The current implementation of the solvers does have some limitations:
- The (2D) finite element mesh needs to be 2D evenly spaced rectangular shaped
- 
does not support using arbitary finite element meshes 



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




