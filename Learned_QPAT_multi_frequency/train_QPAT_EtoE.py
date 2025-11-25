
import numpy as np


import QPAT_util_multifreq as Qutil
import QPAT_nets_multifreq as Nets
import QPAT_FEM_multifreq as FE

import torch
import time
from torch import nn
from torch import optim
import tensorboardX
import scipy.io


FLAGS = None


def training(trainSet,testSet,geom,experimentName,filePath,
             bSize = 1,
             trainIter = 30001,
             LGSiter = 5,
             useTensorboard = True,
             lValInit=1e-4,
             device = 'cuda'):
    
    
    freq_nro = geom["freq_nro"]
    train_samples = len(trainSet.data) 
    test_samples = len(testSet.data)
    
    # Image dimensions

    xsize = geom['xsize']
    ysize =  geom['ysize']
    
    n = torch.tensor(np.array(geom['n'],dtype=np.int64)).to(device)
    solver = geom['solver']
    # Constant part(s) of System matrix (Boundary part)
    B = torch.from_numpy(FE.SysmatComponent('BndPFF', geom)).float().to(device)

    #print("Using L2 loss")
    loss_type = nn.MSELoss()

    print('Using ' + str(LGSiter) + ' end-to-end iterations') 

                
    JJ,det_J_t,indices_elem,node_matrix,node_vector,indice_matrix,M_base =  FE.SysmatComponent_auxiliary(geom,to_pytorch=True,device=device)

    print("Transfering to Device")        
    elem = torch.from_numpy(np.array(geom['elem'],dtype=np.int64)).to(device)
    V_fff_torch = geom['V_fff'].float().to(device)
    V_fdd_torch = geom['V_fdd'].float().to(device)

    
    qvec_torch = torch.from_numpy(geom['qvec']).float().to(device)
    segments = trainSet.segments.int()
    segments_test = testSet.segments.int()
    mus_weight = torch.tensor(geom['mus_weight']).to(device)
    
    geom_torch = Qutil.geom_specs(B, 
                        n, xsize, ysize,elem,V_fff_torch,V_fdd_torch,solver,
                        JJ,det_J_t,indices_elem,node_matrix,node_vector,
                        indice_matrix,M_base=M_base)

    
    print("...completed")
    
    # Training set variables
    stds_train = geom['stds_train'].to(device)
    current_input = trainSet.initial.float()
    C_train = trainSet.C.float()
    priors_train = trainSet.prior_vals.float()
    true_train = trainSet.true
    new_input = torch.zeros(2,train_samples,freq_nro,xsize*ysize,device=device)
    train_writer = tensorboardX.SummaryWriter(comment="/LGS_EtoE_"+str(LGSiter)+"/train")
    data_torch = trainSet.data.float().to(device)



    # Test set variables
    stds_test = geom['stds_test'].to(device)
    C_test = testSet.C.float()
    priors_test = testSet.prior_vals.float()
    current_input_test = testSet.initial.float()
    true_test = testSet.true
    new_input_test = torch.zeros(2,test_samples,freq_nro,xsize*ysize,device=device)    
    validation_writer = tensorboardX.SummaryWriter(comment="/LGS_EtoE_"+str(LGSiter)+"/validation")        
    data_torch_test = testSet.data.float().to(device)

    if geom["crop_water"] == True:
        cropped_indices =  torch.from_numpy(geom['cropped_indices']).to(torch.int64).to(device)
    else:
        cropped_indices = 0


    combined = False    


    with torch.no_grad():
        
        qvec_scaling_file = "normalization_factors/qvec_scaling_factors_nodes_"+str(geom['n'])+"_wavelenghts_"+str(freq_nro)+".pt"
        qvec_scaling = Qutil.Det_norm_factor(qvec_scaling_file,train_samples,freq_nro,
                                             trainSet,device,mus_weight,geom_torch,data_torch,qvec_torch)

        for i in range(len(qvec_scaling)):
            qvec_torch[:,i] = qvec_torch[:,i]*qvec_scaling[i]
            

        model = Nets.LGS_EtoE(loss_type, xsize, ysize,niter=LGSiter,combined=combined,cropped=geom["crop_water"]).to(device)

        optimizer = optim.Adam(model.parameters(), lr=4e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=4e-5,total_steps=trainIter,pct_start=0.1,div_factor=10,final_div_factor=100)

        mua_all = current_input[0].to(device)
        mus_all = current_input[1].to(device)  
        
        mus_weight = mus_weight.to(device)
        
        print("Computing initial gradients (Should take a few minutes...):")

        grad_mua_init,grad_mus_init,grad_mua_init_2 = FE.Get_grad_torch(geom_torch, mua_all, mus_all, data_torch.to(device),segments.to(device),train_samples, 
                                                qvec_torch,C_train.to(device),stds_train.to(device),priors_train.to(device), device)
          
        init_grad_train = torch.cat((grad_mua_init[None],grad_mus_init[None],grad_mua_init_2[None]),dim=0)  
        
        
        mua_all_test = current_input_test[0].to(device)
        mus_all_test = current_input_test[1].to(device)
        
        grad_mua_test,grad_mus_test,grad_mua_test_2 = FE.Get_grad_torch(geom_torch, mua_all_test, mus_all_test, data_torch_test.to(device), segments_test.to(device), test_samples, 
                                         qvec_torch,C_test.to(device),stds_test.to(device),priors_test.to(device), device)

                        
        init_grad_test = torch.cat((grad_mua_test[None],grad_mus_test[None],grad_mua_test_2[None]),dim=0)
           

        sample_info_train = Qutil.sample_info(qvec_torch,stds_train,data_torch,priors_train,C_train,true_train,segments,init_grad_train)
        sample_info_test = Qutil.sample_info(qvec_torch,stds_test,data_torch_test,priors_test,C_test,true_test,segments_test,init_grad_test)
  

        print("Done")
        
        time_forw = 0
        time_bp = 0
        
        
    indices_train = np.array(np.arange(train_samples))
    sample_ind = 0
    epoch = 1
    for it in range(trainIter):
        
        if it % train_samples == 0:
            print("epoch: " + str(epoch))
            epoch+=1
            np.random.shuffle(indices_train)
            sample_ind = 0
        else:
            sample_ind += 1
        
        start = time.time()
        batch = trainSet.next_batch(bSize,indices_train[sample_ind])
        freq_ind = batch[4]
        sample_nro = indices_train[sample_ind]
        batch_f = batch[2].float().to(device)   
        batch_f = torch.cat((batch_f[0,[0]],batch_f[1,[0]],batch_f[0,[1]]),dim=0)
        
        batch_info = sample_info_train.Get_sample_info(sample_nro,freq_ind,device)
        model.train()    
        optimizer.zero_grad()
        
        
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        batch_output,loss = model(geom_torch,batch_f,batch_info,mus_weight,device=device,cropped_indices=cropped_indices,cur_data=0)
        
        time_forw += time.time()-start
        
        start = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()  
        time_bp += time.time()-start
                
        if (~torch.isfinite(batch_output)).any():
            print("Nan or Inf in output")
    
        verbose_every = 500
        if it % verbose_every == 0 and it > 0:
            
            print('it: '+ str(it))
            print("Forward took on average: " + str(time_forw/verbose_every))
            time_forw = 0
            print("Backpropagation took on average: " + str(time_bp/verbose_every))
            time_bp = 0
            print("RAM used (GB): " + str(torch.cuda.max_memory_reserved(device=device)/10**9))
            
        if it % 500 == 0:
            if(useTensorboard):
                with torch.no_grad():    
                    model.eval()
                    if it == 0:
                        monitor_test = 4
                        monitor_train = 40
                        freqs = torch.arange(0,freq_nro)
                        outputs_eval = torch.zeros(2,monitor_train,freq_nro,xsize*ysize).float().to(device)
                        if geom["crop_water"]==True:
                            true_images_eval_whole = geom["orig_images_train"]                               
                        else:
                            true_images_eval_whole = trainSet.true[:,0:monitor_train,:]
                        if geom['test'] == True:
                            out_test_eval = torch.zeros(2,monitor_test,freq_nro,xsize*ysize).float().to(device)
                            if geom["crop_water"]==True:
                                true_im_test_eval_whole = geom["orig_images_test"]                    
                            else:
                                true_im_test_eval_whole = testSet.true[:,0:monitor_test,:]

                    

                    # Evaluated train set

                    output_eval,loss_train = Qutil.Evaluate_set_EtoE(trainSet,xsize,ysize,monitor_train,
                                model,geom_torch,cropped_indices,current_input,sample_info_train,outputs_eval,mus_weight,device)
                    print('Train loss: ' + str(round(loss_train.item(),7)))

                    single_seg_inds = torch.nonzero(trainSet.segments[-1].to(device))

                    Qutil.summaries(train_writer,outputs_eval[:,0:monitor_train,freqs],true_images_eval_whole[:,0:monitor_train,freqs,:].to(device),
                              loss_train, it,xsize,single_seg_inds)
                    
                    if geom['test'] == True:

                        output_eval,loss_validation  = Qutil.Evaluate_set_EtoE(testSet,xsize,ysize,monitor_test,
                                    model,geom_torch,cropped_indices,current_input,sample_info_test,out_test_eval,mus_weight,device)
                        print('Validation loss: ' + str(round(loss_validation.item(),7)))

                        single_seg_inds = torch.nonzero(testSet.segments[-1].to(device))
                        Qutil.summaries(validation_writer,out_test_eval[:,0:4,freqs] ,true_im_test_eval_whole[:,0:4,freqs].to(device),
                                  loss_validation, it,xsize,single_seg_inds)
    
    
    with torch.no_grad(): 
        model.eval()
        print("Updating inputs for next networks")            

        new_input,loss_train = Qutil.Evaluate_set_EtoE(trainSet,xsize,ysize,train_samples,
                    model,geom_torch,cropped_indices,current_input,sample_info_train,new_input,mus_weight,device)


        new_input_test,loss_test = Qutil.Evaluate_set_EtoE(testSet,xsize,ysize,test_samples,
                    model,geom_torch,cropped_indices,current_input_test,sample_info_test,new_input_test,mus_weight,device)

    
    if solver == 0:
        reco_folder = "reconstructions/EtoE_GD/"  
    elif solver == 1:
        reco_folder = "reconstructions/EtoE_GN/"  
             
    
    # Save reconstructions
    out_train = torch.transpose(torch.reshape(new_input,(2,train_samples,freq_nro,xsize,ysize)),3,4).cpu().numpy()
    out_file_path = reco_folder+"train_" + experimentName + "LGS_model_"+ str(LGSiter)+ '_outputs.mat'
    scipy.io.savemat(out_file_path, {'trainout': out_train})
    out_test = torch.transpose(torch.reshape(new_input_test,(2,test_samples,freq_nro,xsize,ysize)),3,4).cpu().numpy()
    out_file_path = reco_folder+"test_" + experimentName + "LGS_model_"+ str(LGSiter)+ '_outputs.mat'
    scipy.io.savemat(out_file_path, {'testout': out_test})
    
    # Save model of current iteration
 
    torch.save(model.state_dict(), filePath + experimentName + "_" + str(LGSiter) + "_of_" + str(LGSiter))

        
    return 0
















