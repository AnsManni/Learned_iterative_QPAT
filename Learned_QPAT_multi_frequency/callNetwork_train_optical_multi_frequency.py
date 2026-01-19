
''' Utility functions for Learned image recontruction 
# Written 2025 by Anssi Manninen, University of Oulu '''

import QPAT_util_multifreq as Qutil
import QPAT_FEM_multifreq as FE
import torch
import os
import pickle

custom_problem = False
project_name = "digital_twin_QPAT"


    #### set paths ####

filePath          = 'TrainedOpticalNets/'   # Where to save the networks
geomPath          = 'geom_files/'           # geometric files
# Coefficients
setname_test_coeffs = "train_sets_multi_frequency/coeffs/experimental_test_set_coeffs_nodes_5184_samples_21.pt" 
setname_train_coeffs = "train_sets_multi_frequency/coeffs/experimental_train_set_coeffs_nodes_5184_samples_21.pt" 
# Datasets
setname_train_data = "train_sets_multi_frequency/data/experimental_train_set_data_nodes_5184_samples_21.pt" 
setname_test_data = "train_sets/data/experimental_test_set_data_nodes_5184_samples_21.pt" 


crop_water          = True                      # Crop water around the circles

if crop_water == True:
    geomfile = 'geom_files/cropped_mesh_data_25_20_nodes_5184.pkl'
else:
    geomfile = 'geom_files/mesh_data_25_20_nodes_5184.pkl'


###############################
####### Main parameters #######
###############################

# Type of step direction ('GN' 'GD' 'SR1' 'UNET')
solver           = 'GN'

# Number of learned iterations (if not using UNET)
LGSiter          = 2

# Training scheme ('greedy' / 'EtoE')
training_scheme  = 'EtoE'

    #### Training related parameters ####

# Training iterations
trainIter        =  50000     # Unet: 80000-100000 | iterative: 35000-50000
    
# Initial learning rate 
lValInit         = 1e-4

# Device ('cuda' 'cpu')
device = 'cuda'

# Weight mua/musp for more balanced learning
mus_weighting     = 12              


    #### Evaluate/visualize ####

# Use Tensorboard for tracking
useTensorboard   = True

# "Test"/"Train"/"both"
testFlag          = True

# Visualize results (?)
visualize_results = False

    ## Advanced options ##


if solver == "Unet":
    crop_water          = False

## DO NOT CHANGE AT THE MOMENT
bSize             = int(1)          # Batch size
prior_adapting    = False           # Adapt prior mean to current optical values
augmented_data    = True            # Flip samples in the training set to double the number
log_data          = True            # Use log(data) 
freqs             = 6               # Nro of used wavelenghts


if training_scheme == "greedy":
        print("Using greedy training")
        import train_QPAT_greedy as Qtrain
else:
        print("Using EtoE training")
        import train_QPAT_EtoE as Qtrain 


print('Using: ' + str(LGSiter) + ' greedy iterations with '+ solver + ' steps')


''' Load samples and geometric details from Matlab files '''

### Load geometric info
with open(geomfile, 'rb') as f:
    geom = pickle.load(f)

# Change indices to start from 0
geom['elem'] = geom['elem']-1
geom['bound_nodes'] = geom['bound_nodes']-1
geom['cropped_indices'] = geom['cropped_indices']-1
n = len(geom["coords"])
nodes = geom['xsize'] * geom['ysize']


data_test = torch.load(setname_test_data) 
data_train = torch.load(setname_train_data) 
coeffs_test = torch.load(setname_test_coeffs) 
coeffs_train = torch.load(setname_train_coeffs) 

samples_train = len(data_train)
samples_test = len(data_test)

intp_train_set = torch.zeros((6,samples_train,freqs,nodes))
intp_train_set[0] = data_train
intp_train_set[1:6] = coeffs_train

intp_test_set = torch.zeros((6,samples_test,freqs,nodes))
intp_test_set[0] = data_test
intp_test_set[1:6] = coeffs_test

# Weight scattering
intp_train_set[2] /= mus_weighting
intp_test_set[2] /= mus_weighting


# Set optical coefficients for water to be constant to get same illumination 
# profile at the boundary of the tube

a = intp_train_set[3].bool()
intp_train_set[1][~a] = 0.01
intp_train_set[2][~a] = 0.0001
intp_train_set[4][~a] = 0.01
intp_train_set[5][~a] = 0.0001

a = intp_test_set[3].bool()
intp_test_set[1][~a] = 0.01
intp_test_set[2][~a] = 0.0001
intp_test_set[4][~a] = 0.01
intp_test_set[5][~a] = 0.0001


if crop_water == True:
    cropped_indices = geom['cropped_indices']
    
    geom["orig_images_train"] = intp_train_set[1:3]
    geom["orig_images_test"] = intp_test_set[1:3]
    intp_test_set = intp_test_set[:,:,:,cropped_indices[:,0]]
    intp_train_set = intp_train_set[:,:,:,cropped_indices[:,0]]

else:
    cropped_indices = torch.arange(n)

# Draw prior values uniformly -25%--25% off from GT
if custom_problem == False:
    intp_train_set, intp_test_set = Qutil.Draw_offset(intp_train_set,intp_test_set,samples_train,samples_test,freqs,intp_train_set[3],intp_test_set[3])

# Log scaled data
if log_data == True and solver !="Unet":
    intp_train_set[0] = torch.log(intp_train_set[0])
    intp_test_set[0] = torch.log(intp_test_set[0])

print('Preparing input data')
''' Prepare input data ''' 
# Generate constant integrals of FEM matrice for efficient gradient calculation
print("Initializing the FE matrices...")

if crop_water== True:
    V_fff_file = "sparseMats/cropped_sparsemat_fff_" + str(geom['xsize']) + "_x_" + str(geom['ysize']) + ".pt"
    V_fdd_file = "sparseMats/cropped_sparsemat_fdd_"+ str(geom['xsize']) + "_x_" + str(geom['ysize']) + ".pt"
else:
    V_fff_file = "sparseMats/sparsemat_fff_" + str(geom['xsize']) + "_x_" + str(geom['ysize']) + ".pt"
    V_fdd_file = "sparseMats/sparsemat_fdd_"+ str(geom['xsize']) + "_x_" + str(geom['ysize']) + ".pt"

if solver != "Unet":
    V_fff,V_fdd = FE.Form_V(geom,V_fff_file,V_fdd_file)



### White-noise prior (same std for each sample) ###
# Prior weights
std_abs  =  0.85 #0.05*data_norm
std_scat =  8.5 #std_abs*10*data_norm


stds_train = torch.ones(2,samples_train,freqs)
stds_train[0] *= std_abs
stds_train[1] *= std_scat

stds_test = torch.ones(2,samples_test,freqs)
stds_test[0] *= std_abs
stds_test[1] *= std_scat


# Computed Wavelength dependency: f(w) = c*exp(-b*w)
b = 0.0016
# Matrice C (S x f x f) indicates the estimated average ratio of scattering values between
# ith and jth wavelength for each sample
C_training = Qutil.Det_freq_dependency(samples_train,freqs,intp_train_set[2],intp_train_set[3,:,0,:],b)
C_test = Qutil.Det_freq_dependency(samples_test,freqs,intp_test_set[2],intp_test_set[3,:,0,:],b)

scaled = False

# Form datasets
trainSet = Qutil.DataSet(intp_train_set[0], intp_train_set[1:3] , intp_train_set[4:6],intp_train_set[3,:,0,:],C_training,intp_train_set[4:6])
testSet = Qutil.DataSet(intp_test_set[0],intp_test_set[1:3], intp_test_set[4:6],intp_test_set[3,:,0,:],C_test,intp_test_set[4:6])
  
# Set attributes geom 
geom['V_fff'] = V_fff
geom['V_fdd'] = V_fdd
geom["stds_train"] = stds_train
geom["stds_test"] = stds_test
geom['mus_weight'] = mus_weighting
geom['bkg_adapting'] = prior_adapting
geom['crop_water'] = crop_water
geom['test'] = testFlag
geom['freq_nro'] = freqs
geom["n"] = n
geom['visualize'] = visualize_results


if  solver == "GD":
    geom['solver'] = 0
elif solver == "GN":
    geom['solver'] = 1
else:
    geom['solver'] = 2


# Make needed folders
if os.path.isdir(filePath) == False:
    os.mkdir(filePath)
if os.path.isdir('reconstructions/') == False:
    os.mkdir('reconstructions/')
if os.path.isdir('runs/') == False:
    os.mkdir('runs/')


   # Name for the experiment
experimentName =  project_name+"_"+training_scheme + '_nodes_'+str(n) + '_' + solver  + '_iter_' + str(LGSiter) + '_trainiter_' + str(trainIter)+"_" 

if training_scheme == "greedy":

    for i in range(LGSiter):
        new_input, new_input_test = Qtrain.training(trainSet,testSet,geom,experimentName,filePath,i,
        bSize = bSize,
        trainIter = trainIter,
        LGSiter = LGSiter,
        useTensorboard = useTensorboard,
        lValInit = lValInit,
        device=device)
        
        trainSet.initial[:,:,:,:] = new_input[:,:,:,cropped_indices[:,0]]
        testSet.initial[:,:,:,:] = new_input_test[:,:,:,cropped_indices[:,0]]

else:
    
    Qtrain.training(trainSet,testSet,geom,experimentName,filePath,
    bSize = bSize,
    trainIter = trainIter,
    LGSiter = LGSiter,
    useTensorboard = useTensorboard,
    lValInit = lValInit,
    device=device)






