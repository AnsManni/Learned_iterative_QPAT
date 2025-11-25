

import QPAT_util_multifreq as Qutil
import QPAT_nets_multifreq as Nets


import numpy as np
import torch
from torch import nn
from torch import optim
import tensorboardX
import scipy.io
import matplotlib.pyplot as plt




def training(trainSet,testSet,geom,experimentName,filePath,
             bSize = 1,
             trainIter = 30001,
             useTensorboard = True,
             lValInit=1e-4,
             device='cuda'):
    
    freq_nro = geom["freq_nro"]
    train_samples = len(trainSet.data) 
    test_samples = len(testSet.data)
    
    # Image dimensions
    xsize = geom['xsize']
    ysize =  geom['ysize']

    loss_type = nn.MSELoss()
    print('Using U-Net')    
    
    model = Nets.UNet(n_in=1, n_out=1).to(device)
    model_mus = Nets.UNet(n_in=1, n_out=1).to(device)

    train_writer = tensorboardX.SummaryWriter(comment="/Unet/train")
    validation_writer = tensorboardX.SummaryWriter(comment="/Unet/validation")        

    optimizer = optim.Adam(model.parameters(), lr=lValInit)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter) 
    optimizer_mus = optim.Adam(model_mus.parameters(), lr=lValInit)
    scheduler_mus = optim.lr_scheduler.CosineAnnealingLR(optimizer_mus, trainIter) 
    

    ####################
    ##### Training #####
    ####################
    
    
    for it in range(trainIter):

        # Draw sample information
        rng = np.random.default_rng()
        sample_num = rng.choice(train_samples ,size=1,replace=False)
        freq_num = rng.choice(6, size=1, replace=False)
        batch_data = trainSet.data[sample_num,freq_num].to(device)
        batch_mua = trainSet.true[0,sample_num,freq_num].to(device)
        batch_mus = trainSet.true[1,sample_num,freq_num].to(device)
        batch_seg = trainSet.segments[sample_num]
        batch_seg_inv = (batch_seg == 0).nonzero().to(device)
        batch_seg = torch.nonzero(batch_seg[0,:]).to(device)
        batch_data_grid = torch.transpose(batch_data.view(1,ysize,xsize),1,2) # to grid
        
        
        ## Forward ##

        model.train() 
        model_mus.train()
        optimizer.zero_grad()
        optimizer_mus.zero_grad()
        out_mus,loss_mus = model_mus(batch_data_grid[:,None,:,:],loss_type,batch_mus,batch_seg,batch_seg_inv)
        out_mua,loss = model(batch_data_grid[:,None,:,:],loss_type,batch_mua,batch_seg,batch_seg_inv)
        
        
        ## Backpropagate ##
        
        loss.backward()
        optimizer.step()
        scheduler.step()  
        loss_mus.backward()
        optimizer_mus.step()
        scheduler_mus.step() 

            

        # Monitor 
        if it % 5000 == 0 or it == trainIter-1:   
            
            print('it: '+ str(it))
            print("RAM used (GB):")
            print(torch.cuda.max_memory_reserved(device=device)/10**9)  
            if it == trainIter - 1:
                monitor_test = test_samples
                monitor_train = train_samples
                outputs_eval = torch.zeros(2,monitor_train,freq_nro,xsize*ysize).float().to(device)
                outputs_test = torch.zeros(2,monitor_test,freq_nro,xsize*ysize).float().to(device)
            else:
                monitor_test = 4
                monitor_train = 40
            
                
                with torch.no_grad():    
                    model.eval()
                    model_mus.eval()
                    loss_train = 0
                    loss_validation = 0
                    outputs_eval = torch.zeros(2,monitor_train,freq_nro,xsize*ysize).float().to(device)
                    outputs_test = torch.zeros(2,monitor_test,freq_nro,xsize*ysize).float().to(device)

                    outputs_eval,loss_train = Qutil.Evaluate_set_Unet(trainSet,monitor_train,freq_nro,outputs_eval,
                                      device,xsize,ysize,model_mus,model,loss)
                    print('Training loss: ' + str(round(loss_train.item(),7)))

                    outputs_test,loss_validation = Qutil.Evaluate_set_Unet(testSet,monitor_test,freq_nro,outputs_test,
                                      device,xsize,ysize,model_mus,model,loss)
                    print('Validation loss: ' + str(round(loss_validation.item(),7)))

                    single_seg_inds = torch.nonzero(trainSet.segments[-1].to(device))
                    if(useTensorboard):
                        Qutil.summaries(train_writer,outputs_eval,trainSet.true.to(device),
                                  loss, it,xsize,single_seg_inds,mus_weight=1)
                        Qutil.summaries(validation_writer,outputs_test[:,0:4],testSet.true[:,0:4,:].to(device),
                                  loss_validation, it,xsize,single_seg_inds,mus_weight=1)

 

    # Save recontructions
    
    reco_folder = "reconstructions/Unet_recos/"
    out_file_path = reco_folder+"train_" + experimentName + "Unet_outputs.mat"
    scipy.io.savemat(out_file_path, {'trainout': outputs_eval.cpu().numpy()})
    out_file_path = reco_folder+"test_" + experimentName + "Unet_outputs.mat"
    scipy.io.savemat(out_file_path, {'testout': outputs_test.cpu().numpy()})

    # save models
    torch.save(model.state_dict(), filePath + experimentName + "_Unet_absorption")
    torch.save(model_mus.state_dict(), filePath + experimentName  + "_Unet_scattering")

        
    return 0


