# Learned iterative model based QPAT solvers (+UNET)

This repository includes codes and datasets used to compute numerical examples for the paper: *"Towards robust quantitative photoacoustic tomography via learned iterative methods"* [arXiv:2510.27487](https://arxiv.org/abs/2510.27487)

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

The **Python** codes provided in this repository implement iterative learned model-based updates based on **gradient descent**, **Gauss-Newton**, and **rank-1-update** methods that can be used to solve two (2D) optical problems of QPAT, the **ideal problem** and the **digital twin problem** described below. The computed reconstrutions represent the optical absorption and reduced scattering parameters of the given domain. The learning task can be selected to be formulated as **greedy** or **end-to-end**. For comparison learned single-step reconstruction method based on **U-Net** architecture is implemented. To solve the optical problem, the light propagation is modeled by using finite element approximation of **diffusion approximation** (DA). See [Implemented solvers](#Solvers) for details of the implementations.


### Ideal problem
<img width="452" height="124" alt="example_ideal" src="https://github.com/user-attachments/assets/9d60d699-bb0a-4e71-8cbc-6c1ff98e2ba1" />


The ideal problem consists of a comprehensive amount of 1250 samples, with inclusions of randomly shaped and located ellipses as shown above. These generated samples with ellipses can be interpreted to mimic, for instance, a cross-section of veins. The used magnitude of optical values produce a highly diffusive region where the overall modeling error from using DA is negligible. This setup is ideal for comparing the convergence and reconstruction accuracy of the implemented methods. 

The photon fluence field of the samples was simulated using [ValoMC](https://inverselight.github.io/ValoMC/) open Monte Carlo software package for **Matlab**. Two separate simulations were performed for each sample. First by illuminating the domain from the top side of the rectangular domain and secondly from the right side.


### Digital twin problem
<img width="290" height="160" alt="mittaus" src="https://github.com/user-attachments/assets/54149b51-a7af-4112-9a91-30fbc08c6ea5" />


The digital twin problem is much closer to experimental conditions compared to the ideal problem. In this setup the phantoms were immersed in a water bath that was illuminated by 5 sources giving angular coverage of 270 degrees as shown in above figure.

The problem now has limited training data, samples with varying locations of the tubes (region of interest), a large range of possible optical values, and a large modeling error. The digital twin problem re-used data from physical tissue-mimicking phantoms with piecewise-constant material distributions cite{}

The total number of actual digital twin samples for training and testing was only 14. As such, we supplemented the digital twins with additional phantom simulations, where the optical properties are not in correspondence with physical phantoms but instead a random mix of the existing physical phantoms. Each phantom was drawn to have between 1 and 3 inclusions and a radius of 14.2\,mm. We simulated at six wavelengths: 700, 740, 780, 820, 860, and 900 (nm). In total, we supplemented the training data with 41 of these phantom simulations. As the photon fluence was approximately symmetrical the sample set was augmented by vertically flipping each sample doubling number of samples. The fluence simulations were done by using MCX software.

The digital twin problem introduces several sources of modeling error. Firstly, the simulations were conducted in 3D, but we approximated the light propagation in 2D, where the photons need to travel a much shorter path to reach the center. Secondly, the optical values of the samples contained relatively high absorption coefficients, violating assumptions of the DA, hence causing it to be inaccurate.

## Overview of the codes
The codes in this repository are used to train learned iterative model based solvers or (single step) U-Net to solve two different optical problems of QPAT.
The light propagation model is implemented using 2D finite element approximation of **diffusion approximation** (DA). Computing the light fluence from the DA, solving the selected step direction and training the networks are all implemented using PyTorch library allowing use of GPU computation. The used PyTorch functions support [automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) meaning that there's no need to explicitly write the gradients of each operation with respect to the network parameters.

### Finite element approximation of diffusion approximation
The implemented diffusion approximation considers a discretization of the domain with $P$ non-overlapping triangular elements and $N$ grid coordinates. The optical parameters are represent in this discretization with piece-wise continuous basis functions. For detailed representation of the finite element matrices see e.g., cite(X). 

For the **ideal problem** the light source is implemented to uniformly illuminate the top or right boundary elements of the rectangular domain. For the 
**digital twin problem** five 6.12\, mm wide light sources were approximated to have a Gaussian intensity profile with a standard deviation of 3. As there was water between the imaged phantom tube and the light sources, the absorption of the light in the water was computed (different for each wavelength) yielding intensity profiles for the finite element mesh boundary elements. However, for computational conveniency a single finite element mesh was used for all of the samples, even the size and location of the (non-water) phantom tubes slightly varied. This resulted a few layers of water remaining inside the finite element mesh for most of the samples as shown below.

<img width="220" height="216" alt="fe_mesh_twin" src="https://github.com/user-attachments/assets/299fad69-f268-4276-9eb0-06f0c2bbe1d2" />


### Solvers 

**Learned iterative model-based solvers** <br />
The main implementation of this repository are the learned iterative solvers. These iterative solvers consist of $K$ number of networks based on the residual network ResNet architecture. Each of the networks update the current absorption and (reduced) scattering estimates, based on the current optical values and their respective gradient (model) information computed from the used variational problem. 
The codes implement three options for the gradient information:
- **Gradient descent**
- **Gauss-Newton**
- **rank-1-update**

Note: To computed the Gauss-Newton direction robustly for the non-linear ill-posed optical problem, a white-noise prior is applied.

The training task for the $K$ updating networks can be formulated as:
- **Greedy**:The $K$ number of networks are trained via $K$ separate training task
  - Lower memory requirement and faster training as photon fluence and gradient information are computed outside of the training
  - Might lead to less optimal network weights
- **End-to-end**: The $K$ number of networks are trained jointly via single learning task
  - In theory the greedy scheme provides an upper bound on the minimized loss function for end-to-end networks
  - High memory consumption and run time due to photon fluence and gradients being computed $K$ times for each training iteration


**U-Net** <br />
In addition to iterative learned solver, we provide implementation of fully-learned single step recontruction method based on the residual U-Net architecture.
The input for the U-Net is the data, absorbed energy density, from which it directly produced the absorbtion and scattering.


## How to setup
To succesfully run the training scripts follow these steps:

**1.) Setup python environment** <br />
    - via Anaconda <br />
    or <br />
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
Ideal problem (see README in `/Learned_QPAT_multi_illumination`) <br />
or <br />
Digital twin problem (see README in `/Learned_QPAT_multi_frequency`) 




## Extendability of the codes 
The codes are not extensively tested with different measurement geometry inputs and can fail to work with different geometry configurations.  
However, we are happy to help extending the usage of the codes. For queries on possible problems or suggestions for extensions contact anssi.manninen@oulu.fi

**Limitations of the implementation:**
- As in this work we use convolutional networks and not graph convolutional networks, the (2D) finite element mesh is assumed to be 2D evenly spaced rectangular
-


### Input formats:

**Geometry file** <br />
Basic (.pkl file) form of the geometric file that can be directly used contains fields (suggested datatypes that have verified to work):
- 'xsize',     number of vertical nodes                              (int)
- 'ysize',      number of horizontal nodes                            (int)
- 'coords', $(n \times 2)$ vector containing the exact coordinates (in mm) of each node. (NumPy array)
- 'elem',   $(E \times 3)$ vector providing indices of nodes for each triangular element. $E$ is the total number of elements.  (int)
- 'bound_nodes', $(B \times 2)$ vector containing indices of boundary nodes for each boundary element (total of $B$) (NumPy int array)
- 'qvec',       $(nI \times w)$ vector describing the illumination intensity each node receives (i.e., for boundary illumination only the (boundary) node indices that are illuminated are non zero. Repeated for each illumination $I$.  (NumPy array)

 Find the exemplar geom files from `/Learned_QPAT_multi_illumination/geom_files/` and `/Learned_QPAT_multi_frequency/geom_files/`. 
 
**Data file (Absorbed energy density)** <br />
Multi illumination: $(S \times nI)$ vector where $S$ is the number of samples and $I$ number of illuminations. <br />
Multi wavelenght:  $(S \times nI)$ vector where $S$ is the number of samples and $I$ number of illuminations. <br />

**Optical coefficients** <br />
Multi illumination: $(S \times 2 \times n)$ vector where $S$ is the number of samples. 

The first index of second dimension contains the absorption and second reduced scattering  <br />

The current file format for data and coefficients is .pt 






