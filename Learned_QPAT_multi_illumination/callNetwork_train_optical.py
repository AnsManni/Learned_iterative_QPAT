
import QPAT_util as Qutil

import numpy as np
import scipy.io as sio
import pickle

# Make needed folders

#############################
#### Training parameters ####
#############################


solver           = 'GD'      # 'GN' 'GD' 'SR1'
LGSiter          = 2         # Number of trained iterative networks
training_scheme  = 'greedy'  # 'greedy' / 'EtoE'
trainIter        = 3500     # Training iterations
# Initial learning rate    
lValInit         = 2e-4      #  (With many EtoE updates smaller rate might be required)    
 

#### Other changeable parameters ####

interpolate_data = True      # Use smaller FE mesh and interpolate data   
log_data         = True      # log scaled data space
log_sol          = True      # Log scaled solution space


# Use Tensorboard for tracking
useTensorboard   = True      
# Device
device = 'cuda'


if training_scheme == "greedy":
        print("Using greedy training")
        import Train_optical_torch_greedy as PAT
else:
        print("Using end-to-end training")
        import Train_optical_torch_EtoE as PAT 


    #######################################
    #### MODEL AND TRAINING PARAMETERS ####
    #######################################

filePath          = 'TrainedOpticalNets/'   # Where to save the networks
testFlag          = True                    # "Test"/"Train"/"both" model      
bSize             = int(1)
lossFunc          = 'l2_loss'
mcmc_samples      = 1250                    # MCMC samples
ns                = 2                       # Number of sources

   # Name for the experiment
if log_data == True:
    experimentName =  training_scheme + '_opticaltest' + '_' + solver + '_log_scaled_' + '_iter_' + str(LGSiter) + '_trainiter_' + str(trainIter) + '_samples_' + str(mcmc_samples)+ "_" + "large"   #Name for this experiment
else:
    experimentName =  training_scheme + '_opticaltest' + '_' + solver + '_iter_' + str(LGSiter) + '_trainiter_' + str(trainIter) + '_samples_' + str(mcmc_samples)+ "_" + "large"   #Name for this experiment

''' Load samples and geometric details from pickel file '''

# Geometric (Finit element) details

if interpolate_data == True:
    geomfile = 'geom_files/mesh_data_25_20_nodes_2520.pkl'
    geomfile_orig = 'geom_files/mesh_data_25_20_nodes_4636.pkl'
    with open(geomfile_orig, 'rb') as f:
        geom_orig = pickle.load(f)
else:
    geomfile = 'geom_files/mesh_data_25_20_nodes_4636.pkl'

            
with open(geomfile, 'rb') as f:
    geom = pickle.load(f)

n = geom['n']

xsize = geom['xsize']
ysize = geom['ysize']

# Load data and ground truth

sample_file = "train_sets_multi_illumination/images/ValoMC_limited_view_large_images_samples_1250.mat"
images = sio.loadmat(sample_file)['images']

data_file = "train_sets_multi_illuminatio/data/ValoMC_limited_view_noisy_large_dataset_samples_1250.mat"
data = sio.loadmat(data_file)['train_data']

# Interpolate / load interpolated set
if interpolate_data == True:
   setname = 'train_sets_multi_illumination/interpolated_sets/intp_set_2520.pt'
   data,images = Qutil.Interpolate_optical(data,images[:,0,:],images[:,1,:],geom,geom_orig['coords'],setname)      
   

### Add normal distributed noise ###
print('Preparing input data')

### Data split ration (to train and test) ###
samples = len(images[:,0,0])
train_rat = 0.8 # Train-Test/Validation ratio 
indxx = int(np.floor(train_rat*samples))


if log_data == False:
    # Set noise level
    noiseLev = 0.01 
    geom['Le_train'] = np.zeros((indxx,n*ns))
    geom['Le_test'] = np.zeros((samples-indxx,n*ns))
    
    for idx in range(samples):

            if idx < indxx:
                geom['Le_train'][idx,:] = 1/(data[idx,:]*noiseLev)
            else:
                geom['Le_test'][idx-indxx,:] = 1/(data[idx,:]*noiseLev)
                

samples = len(images[:,0,0])

geom['grid_coord'] = geom['grid_coord']-1
grid_coord = geom['grid_coord']

imSize = geom['grid_coord'].shape
print(samples, imSize)

### Scale/normalize data space ###
if log_data == True:
    train_data = np.log(data[0:indxx,:])
    test_data = np.log(data[indxx:samples,:])
else:    
    train_data = data[0:indxx,:]
    test_data = data[indxx:samples,:]

train_true = images[0:indxx,:,:]
test_true = images[indxx:samples,:,:]

# Set initial absorption and scattering
geom['bkg_mua'] = 0.01
geom['bkg_mus'] = 2

init_val_train = np.ones(train_true.shape)
init_val_train[:,0,:] *= geom['bkg_mua']
init_val_train[:,1,:] *= geom['bkg_mus'] 

init_val_test = np.ones(test_true.shape)
init_val_test[:,0,:] *= geom['bkg_mua']
init_val_test[:,1,:] *= geom['bkg_mus'] 

# Transform optical coefficients to grid 

train_true = np.transpose(np.reshape(train_true, (len(train_true),2,ysize,xsize)),(0,1,3,2))
test_true = np.transpose(np.reshape(test_true, (len(test_true),2,ysize,xsize)),(0,1,3,2))
init_val_train = np.transpose(np.reshape(init_val_train, (len(init_val_train),2,ysize,xsize)),(0,1,3,2))
init_val_test = np.transpose(np.reshape(init_val_test, (len(init_val_test),2,ysize,xsize)),(0,1,3,2))


#### FORM WHITE NOISE PRIORS ####
if solver == 'GN':
    STD_att = 0.25*np.max(np.abs(train_true[:,0,:]-geom['bkg_mua']))  # !!
    STD_scat = 0.25*np.max(np.abs(train_true[:,1,:]-geom['bkg_mus']))   #!!
    if log_data:
        STD_att *= 10
        STD_scat *= 10
    geom['Lmua'] = np.diag(np.ones(n)/STD_att)
    geom['Lmus'] = np.diag(np.ones(n)/STD_scat)
else:
    geom['Lmua'] = np.diag(np.ones(2))
    geom['Lmus'] = np.diag(np.ones(2))
    

# Weight abs/scat to balance loss
train_true[:,0,:,:] = train_true[:,0,:,:]/geom['bkg_mua']
train_true[:,1,:,:] = train_true[:,1,:,:]/geom['bkg_mus']
test_true[:,0,:,:] = test_true[:,0,:,:]/geom['bkg_mua']
test_true[:,1,:,:] = test_true[:,1,:,:]/geom['bkg_mus']


# Scale solution space
if log_sol == True: 
    init_val_train[:,1,:,:] = np.log(init_val_train[:,1,:,:]/geom['bkg_mus'])
    init_val_train[:,0,:,:] = np.log(init_val_train[:,0,:,:]/geom['bkg_mua'])   
    init_val_test[:,1,:,:] = np.log(init_val_test[:,1,:,:]/geom['bkg_mus'])
    init_val_test[:,0,:,:] = np.log(init_val_test[:,0,:,:]/geom['bkg_mua'])

# Form datasets
trainSet = Qutil.DataSet(train_data, train_true , init_val_train)
testSet = Qutil.DataSet(test_data,test_true,init_val_test)


# Load sparse FE matrices
V_fff_file = "sparseMats/sparsemat_fff_" + str(n)  + ".pt"
V_fdd_file = "sparseMats/sparsemat_fdd_" + str(n) + ".pt"

V_fff,V_fdd = Qutil.Form_V(geom,V_fff_file,V_fdd_file)

# Placeholders for greedy SR1
if solver == "SR1" and training_scheme == "greedy":
    prev_train = Qutil.Prev_values(LGSiter,n,len(init_val_train),xsize,ysize)
    prev_test = Qutil.Prev_values(LGSiter,n,len(init_val_test),xsize,ysize)
    prev_train.prev_x[0] = init_val_train
    prev_test.prev_x[0] = init_val_test


# Set attributes geom 
geom['solver'] = solver
geom['log_sol'] = log_sol
geom['log_data'] = log_data
geom['test'] = testFlag
geom['ns'] = ns
geom['V_fff'] = V_fff
geom['V_fdd'] = V_fdd


if training_scheme == "greedy":
    for i in range(LGSiter):
        if solver == "SR1":
            new_input, new_input_test,grad_prev,grad_prev_test = PAT.training_greedy(trainSet,testSet,geom,experimentName,filePath,
             lossFunc = lossFunc,
             bSize = bSize,
             trainIter = trainIter,
             LGSiter = LGSiter,
             useTensorboard = useTensorboard,
             lValInit = lValInit,
             currentLGSit=i,
             prev_train=prev_train,
             prev_test=prev_test,
             device=device)
            
        else:
            new_input, new_input_test = PAT.training_greedy(trainSet,testSet,geom,experimentName,filePath,
             lossFunc = lossFunc,
             bSize = bSize,
             trainIter = trainIter,
             LGSiter = LGSiter,
             useTensorboard = useTensorboard,
             lValInit = lValInit,
             currentLGSit=i,
             device=device)
        trainSet.initial[:,:,:,:] = new_input
        testSet.initial[:,:,:,:] = new_input_test
        
        if solver == "SR1":
            prev_train.prev_x[i+1] = new_input
            prev_train.prev_grad[i] = grad_prev
            prev_test.prev_x[i+1] = new_input_test
            prev_test.prev_grad[i] = grad_prev_test
        
else:
    PAT.training_EtoE(trainSet,testSet,geom,training_scheme,experimentName,filePath,
         lossFunc = lossFunc,
         bSize = bSize,
         trainIter = trainIter,
         LGSiter = LGSiter,
         useTensorboard = useTensorboard,
         lValInit = lValInit)
       
    





