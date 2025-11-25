

import QPAT_util_multifreq as Qutil
import QPAT_nets_multifreq as Nets
import QPAT_FEM_multifreq as FE

import numpy as np
import torch
import time
from torch import nn
from torch import optim
import tensorboardX
import scipy.io
import os


def training(trainSet,testSet,geom,experimentName,filePath,currentlgsit,
             bSize = 1,
             trainIter = 30001,
             LGSiter = 5,
             useTensorboard = True,
             lValInit=1e-4,
             device='cuda'):
    
    print('Current iteration: ' + str(currentlgsit+1))    
    lgsit = currentlgsit
    mus_weight = geom['mus_weight']
    xsize = geom['xsize']
    ysize =  geom['ysize']
    n = torch.tensor(np.array(geom['n'],dtype=np.int64)).to(device)
    freq_nro = geom["freq_nro"]
    loss_type = nn.MSELoss()
    cropped_indices =  torch.from_numpy(geom['cropped_indices']).to(torch.int64).to(device)
    
    print("Transfering variables to chosen device")
    # Constant part(s) of System matrix (Boundary part)
    B = torch.from_numpy(FE.SysmatComponent('BndPFF', geom)).float().to(device)
    
    # Auxiliary variables
    V_fff_torch = geom['V_fff'].float().to(device)
    V_fdd_torch = geom['V_fdd'].float().to(device)
    JJ,det_J_t,indices_elem,node_matrix,node_vector,indice_matrix,M_base =  FE.SysmatComponent_auxiliary(geom,to_pytorch=True,device=device)
    elem = torch.from_numpy(np.array(geom['elem'],dtype=np.int64)).to(device)       
    qvec_torch = torch.from_numpy(geom['qvec']).float().to(device)
    solver = geom['solver']
    combined = False    
    
    geom_torch = Qutil.geom_specs(B, 
                        n, xsize, ysize,elem,V_fff_torch,
                        V_fdd_torch,solver,JJ,det_J_t,indices_elem,node_matrix,node_vector,indice_matrix,M_base)
    
    # Training variables
    train_samples,current_input,C_training,priors_train,data_torch,segments = Qutil.Tranfer_to_device(trainSet,device)
    stds_train = geom['stds_train'].to(device)
    new_input = torch.zeros(2,train_samples,freq_nro,xsize*ysize,requires_grad=False).to(device)
    if useTensorboard == True:
        train_writer = tensorboardX.SummaryWriter(comment="/LGS"+str(lgsit)+"/train")


    # Test variables
    test_samples,current_input_test,C_test,priors_test,data_torch_test,segments_test = Qutil.Tranfer_to_device(testSet,device)
    stds_test = geom['stds_test'].to(device)
    new_input_test = torch.zeros(2,test_samples,freq_nro,xsize*ysize,requires_grad=False).to(device)   
    if useTensorboard == True:
        validation_writer = tensorboardX.SummaryWriter(comment="/LGS"+str(lgsit)+"/validation")        

    print("...completed")
        

    # Normalize fluence for DA
    
    with torch.no_grad():

        qvec_scaling_file = "normalization_factors/qvec_scaling_factors_nodes_"+str(geom['n'])+"_wavelenghts_"+str(freq_nro)+".pt"
        qvec_scaling = Qutil.Det_norm_factor(qvec_scaling_file,train_samples,freq_nro,
                                             trainSet,device,mus_weight,geom_torch,data_torch,qvec_torch)
        for i in range(len(qvec_scaling)):
            qvec_torch[:,i] *= qvec_scaling[i]        



        ##############################
        ## Computed step directions ##
        ##############################

        print("Computing step directions...")

        mua_all = current_input[0]
        mus_all = current_input[1]    
        if geom['bkg_adapting'] == True and lgsit > 0:
            priors_train = torch.cat((mua_all[None].detach().clone(),mus_all[None].detach().clone()),dim=0)
        start = time.time()

        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #    with record_function("gradients"):
        dx_mua,dx_mus,dx_mua_2 = FE.Get_grad_torch(geom_torch, mua_all, mus_all, data_torch,segments,train_samples, 
                                qvec_torch,C_training,stds_train,priors_train, device)
                
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("Took (on average): ", str((time.time()-start)/train_samples/3))

        current_dx = torch.cat((dx_mua[None],dx_mus[None],dx_mua_2[None]),dim=0)  

        print("...also for test samples...")

        # Test set

        mua_all_test = current_input_test[0]
        mus_all_test = current_input_test[1]

        if geom['bkg_adapting'] == True  and lgsit > 0:
            priors_test = torch.cat((mua_all_test[None].detach().clone(),mus_all_test[None].detach().clone()),dim=0)


        dx_mua_test,dx_mus_test,dx_mua_test_2 = FE.Get_grad_torch(geom_torch, mua_all_test, mus_all_test, data_torch_test, segments_test, test_samples, 
                                    qvec_torch,C_test,stds_test,priors_test, device)
        

                    
        current_dx_test = torch.cat((dx_mua_test[None],dx_mus_test[None],dx_mua_test_2[None]),dim=0)
   
        ## Initialize model and optimizer
        model = Nets.LGS(loss_type, xsize, ysize,combined=combined,cur_it=lgsit,cropped=geom["crop_water"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lValInit)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter) 

    time_forw = 0
    time_bp = 0
    sample_ind = 0
    epoch = 1
    indices_train = np.array(np.arange(train_samples))

    for it in range(trainIter):
                      
        if it % train_samples == 0:
            #print("epoch: " + str(epoch))
            epoch+=1
            np.random.shuffle(indices_train)
            sample_ind = 0
        else:
            sample_ind += 1
        
        start = time.time()
        batch = trainSet.next_batch(bSize,indices_train[sample_ind])
            
        batch_images = batch[1].float().to(device)
        batch_images = torch.cat((batch_images[0,[0]],batch_images[1,[0]],batch_images[0,[1]],batch_images[1,[1]]),dim=0)
        batch_f = batch[2].float().to(device)   
        batch_f = torch.cat((batch_f[0,[0]],batch_f[1,[0]],batch_f[0,[1]]),dim=0)
        batch_segments = batch[3].float().to(device)
        freq_ind = batch[4]
        sample_nro = indices_train[sample_ind]
        batch_grad = torch.squeeze(current_dx[:,sample_nro,int(freq_ind/2)])
        C_batch = C_training[sample_nro,freq_ind,freq_ind+1]

        ##  Forward 
        start = time.time()
        
        model.train()    
        optimizer.zero_grad()
        batch_output,loss = model(batch_f, batch_grad,batch_segments, batch_images,mus_weight,C_batch,
                        device=device,cropped_indices = cropped_indices,cur_data=0)

        time_forw += time.time()-start

        ## Training  step
        
        start = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()  
        time_bp += time.time()-start
    

        # Evaluate and monitor
        if it % 500 == 0:
            print('it: '+ str(it))
            print("Forward took on average: " + str(time_forw/500))
            print("Backpropagation took on average: " + str(time_bp/500))
            print("RAM used (GB): " + str(torch.cuda.max_memory_reserved(device=device)/10**9))
            time_bp = 0
            time_forw = 0

            # Monitor with Tensorboard
            if(useTensorboard):
                
                    with torch.no_grad():    
                        model.eval()
                        if it == 0:
                            
                            # Monitor part of samples
                            monitor_test = 4
                            monitor_train = train_samples
                            freqs = torch.arange(0,freq_nro)
                            
                            
                            outputs_eval = torch.zeros(2,monitor_train,freq_nro,xsize*ysize).float().to(device)
                            if geom["crop_water"]==True:
                                true_images_eval_whole = geom["orig_images_train"]                               
                            else:
                                true_images_eval_whole = trainSet.true[:,0:monitor_train,:]
                            true_images_eval = trainSet.true[:,0:monitor_train,:]
       
            
                            if geom['test'] == True:
                                out_test_eval = torch.zeros(2,monitor_test,freq_nro,xsize*ysize).float().to(device)
                                if geom["crop_water"]==True:
                                    true_im_test_eval_whole = geom["orig_images_test"]                    
                                else:
                                    true_im_test_eval_whole = testSet.true[:,0:monitor_test,:]
                                true_im_test_eval = testSet.true[:,0:monitor_test,:]
           


                        outputs_eval, loss_train = Qutil.Evaluate_set_greedy(trainSet,model,device,
                                        monitor_train,freqs,cropped_indices,outputs_eval,
                                        current_input,true_images_eval,C_training,mus_weight,current_dx)
                        print('Train loss: ' + str(round(loss_train.item(),7)))


                        single_seg_inds = torch.nonzero(trainSet.segments[-1].to(device))
                        Qutil.summaries(train_writer,outputs_eval[:,0:monitor_train,freqs],true_images_eval_whole[:,:,freqs,:].to(device),
                                  loss_train,it,xsize,single_seg_inds,cropped_indices=cropped_indices,mus_weight=mus_weight)
                        
                        if geom['test'] == True:
                            out_test_eval, loss_validation = Qutil.Evaluate_set_greedy(testSet,model,device,
                                            monitor_test,freqs,cropped_indices,out_test_eval,
                                            current_input_test,true_im_test_eval,C_test,mus_weight,current_dx_test)
                            print('Validation loss: ' + str(round(loss_validation.item(),7)))

                            single_seg_inds=torch.nonzero(testSet.segments[-1].to(device))
                            Qutil.summaries(validation_writer,out_test_eval[:,0:monitor_test,freqs] ,true_im_test_eval_whole[:,0:4,freqs].to(device),
                                      loss_validation, it,xsize,single_seg_inds,cropped_indices=cropped_indices,mus_weight=mus_weight)
                    
                
    with torch.no_grad():    
        model.eval()

        print("Updating inputs for next networks")            


        new_input, loss_train = Qutil.Evaluate_set_greedy(trainSet,model,device,
                        train_samples,freqs,cropped_indices,new_input,
                        current_input,trainSet.true.to(device),C_training,mus_weight,current_dx)
        
        
        new_input_test, loss_validation = Qutil.Evaluate_set_greedy(testSet,model,device,
                        test_samples,freqs,cropped_indices,new_input_test,
                        current_input_test,testSet.true.to(device),C_test,mus_weight,current_dx_test)

        
        
        # Make new folder for the current run
        
        if solver == 0:
            str_solver =  "GD"  
        elif solver == 1:
            str_solver =  "GN"  
            
        if geom['visualize'] == True:
            vis_recos = torch.reshape(new_input[0:2,0,0],(1,2,xsize,ysize)).cpu().numpy()
            vis_true = torch.reshape(geom["orig_images_train"][0:2,0,0],(1,2,xsize,ysize)).cpu().numpy()
            name = "example_images/train_reco_greedy_"+str_solver+"_"+str(lgsit)+".png"
            Qutil.Visualize_samples(vis_true,name,'Ground truth',mus_weight)
            Qutil.Visualize_samples(vis_recos,name,'Estimates')

            vis_recos = torch.reshape(new_input_test[0:2,0,0],(1,2,xsize,ysize)).cpu().numpy()
            vis_true = torch.reshape(geom["orig_images_test"][0:2,0,0],(1,2,xsize,ysize)).cpu().numpy()
            name = "example_images/test_reco_greedy_"+str_solver+"_"+str(lgsit)+".png"
            Qutil.Visualize_samples(vis_true,name,'Ground truth',mus_weight)
            Qutil.Visualize_samples(vis_recos,name,'Estimates')
        
        run_nro = 0

        while os.path.isdir("reconstructions/greedy_run_"+str_solver+"_"+str(run_nro)+"/"):
            run_nro +=1
        
        if lgsit == 0:
            reco_folder = "reconstructions/greedy_run_"+str_solver+"_"+str(run_nro)+"/"
            os.mkdir(reco_folder)
        else:
            reco_folder = "reconstructions/greedy_run_"+str_solver+"_"+str(run_nro-1)+"/"

        # Save reconstructions (in grid form)
        

        new_input = new_input.detach().cpu()
        new_input_test = new_input_test.detach().cpu()
        
        out_train = torch.transpose(torch.reshape(new_input,(2,train_samples,freq_nro,xsize,ysize)),3,4).cpu().numpy()
        out_file_path = reco_folder+"train_" + experimentName + "LGS_model_"+ str(lgsit)+ '_outputs.mat'
        scipy.io.savemat(out_file_path, {'trainout': out_train})
        out_test = torch.transpose(torch.reshape(new_input_test,(2,test_samples,freq_nro,xsize,ysize)),3,4).cpu().numpy()
        out_file_path = reco_folder+"test_" + experimentName + "LGS_model_"+ str(lgsit)+ '_outputs.mat'
        scipy.io.savemat(out_file_path, {'testout': out_test})
        
    
        # Save model
        torch.save(model.state_dict(), filePath + experimentName + "_" + str(lgsit+1) + "_of_" + str(LGSiter))
        
        
  
        return new_input.cpu().detach(), new_input_test.cpu().detach()
            

