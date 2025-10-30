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

The **Python** codes provided in this repository implement iterative learned model-based updates based on **gradient descent**, **Gauss-Newton**, and **rank-1-update** methods that can be used to solve two (2D) optical problems of QPAT, the **ideal problem** and the **digital twin problem** described below. The computed reconstrutions represent the optical absorption and reduced scattering parameters of the given domain. The learning task can be selected to be formulated as **greedy** or **end-to-end**. For comparison learned single-step reconstruction method based on **U-Net** architecture is implemented. To solve the optical problem, the light propagation is modeled by using finite element approximation of **diffusion approximation** (DA). See [Implemented solvers](#Implemented-solvers) for details of the implementations.


### Ideal problem
The ideal problem consists of a comprehensive amount of 1250 samples, with inclusions of randomly shaped and located ellipses. These generated samples with ellipses can be interpreted to mimic, for instance, a cross-section of veins. The used magnitude of optical values produce a highly diffusive region where the overall modeling error from using DA is negligible. This setup is ideal for comparing the convergence and reconstruction accuracy of the implemented methods. 

The photon fluence field of the samples was simulated using ValoMC (https://inverselight.github.io/ValoMC/) open Monte Carlo software package for **Matlab**. Two separate simulations were performed for each sample. First by illuminating the domain from top side of the rectangular domain and secondly from right side.


### Digital twin problem
The digital twin problem is much closer to experimental conditions compared to the ideal problem. The problem now has limited training data, samples with varying locations of the tubes (region of interest), a large range of possible optical values, and a large modeling error. The digital twin problem re-used data from physical tissue-mimicking phantoms with piecewise-constant material distributions cite{}

The total number of actual digital twin samples for training and testing was only 14. As such, we supplemented the digital twins with additional phantom simulations, where the optical properties are not in correspondence with physical phantoms but instead a random mix of the existing physical phantoms. Each phantom was drawn to have between 1 and 3 inclusions and a radius of 14.2\,mm. We simulated at six wavelengths: 700, 740, 780, 820, 860, and 900 (nm). In total, we supplemented the training data with 41 of these phantom simulations. As the photon fluence was approximately symmetrical the sample set was augmented by vertically flipping each sample doubling number of samples. 

The digital twin problem introduces several sources of modeling error. Firstly, the simulations were conducted in 3D, but we approximated the light propagation in 2D, where the photons need to travel a much shorter path to reach the center. Secondly, the optical values of the samples contained relatively high absorption coefficients, violating assumptions of the DA, hence causing it to be inaccurate.

## Overview of the codes
The codes in this repository are used to train learned iterative model based solvers or (single step) U-Net to solve two different optical problems of QPAT.
The light propagation model is implemented using 2D finite element approximation of **diffusion approximation** (DA). Computing the light fluence from the DA, solving the selected step direction and training the networks are all implemented using PyTorch library allowing use of GPU computation. The used PyTorch functions support [automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) meaning that there's no need to explicitly write the gradients of each operation with respect to the network parameters.

### Finite element approximation of diffusion approximation
The implemented diffusion approximation considers a uniform discretization of the domain with $P$ non-overlapping triangular elements and $N$ grid coordinates. The optical parameters are represent in this discretization with piece-wise continuous basis functions. For the **ideal problem** the light source is implemented to uniformly illuminate the top or right boundary elements of the rectangular domain. For the 

### Solvers 

**Learned iterative model-based solvers** <br />
- Greedy / end-to-end training 
- Gradient descent / Gauss-Newton / rank-1-update direction used as the information for the networks <br />

**U-Net** <br />
(single step) U-Net <br />

**Changeable parameters**


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




## Extendability of the codes 
The codes are not extensively tested with different measurement geometry inputs and hence can fail to properly work with different geometry files.  <br />
However, we are happy to help extending the usage of the codes. For queries on possible problems or suggestions for extensions contact anssi.manninen@oulu.fi

**The current implementation of the solvers does have some limitations:**
- As in this work we use convolutional networks and not graph convolutional networks, the (2D) finite element mesh is assumed to be 2D evenly spaced rectangular
-


### Input formats:

**Geometry file** (/geom_files/ and /geom_files/)
Basic form the the geometric file contains fields (only the suggested datatypes have verified to work):
- 'xsize',     number of vertical nodes                              (uint8)
- 'ysize',      number of horizontal nodes                            (uint8)
- 'n',          number of total nodes                                 (int32)    
- 'qvec',       $(n*I \times 1)$ vector describing the illumination intensity each node receives (i.e., for boundary illumination only the (boundary) node indices that are illuminated are non zero. Repeated for each illumination $I$.  (double)
- 'elem',   () describing element indices of each element  
- 'bound_nodes',
- 'coords',









