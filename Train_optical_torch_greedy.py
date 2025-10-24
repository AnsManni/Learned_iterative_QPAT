''' Class file using pytorch Learned image recontruction for optical problem of QPAT
Written 2025 by '''

import numpy as np

import QPAT_util as Qutil
import QPAT_nets as Nets

import torch
import time
from torch import nn
from torch import optim
import tensorboardX
import scipy.io
import os


def training_greedy(dataSet,testSet,geom,experimentName,filePath,
             lossFunc = 'l2_loss',
             bSize = 1,
             trainIter = 35000,
             LGSiter = 4,
             useTensorboard = True,
             lValInit=2e-4,
             currentLGSit=0,
             prev_train=0,
             prev_test=0,
             device = 'cuda'):
    
    print("Transfering to Device")        

    lgsit = currentLGSit

    # Image dimensions
    xsize = geom['xsize']
    ysize =  geom['ysize']
    
    n = geom['n']

    # Constant part(s) of System matrix (Boundary part)
    B = torch.from_numpy(Qutil.SysmatComponent('BndPFF', geom)).float().to(device)

    loss = nn.MSELoss()

    print('Using ' + str(LGSiter) + ' greedy iterations')
        
    V_fff_torch = geom['V_fff'].float().to(device)
    V_fdd_torch = geom['V_fdd'].float().to(device)
                
    JJ,det_J_t,node_matrix,node_vector,indice_matrix =  Qutil.SysmatComponent_auxiliary(geom)

    indice_matrix = indice_matrix.float().to(device)
    JJ = torch.from_numpy(JJ).float().to(device)
    node_matrix = torch.from_numpy(node_matrix).float().to_sparse().to(device)
    node_vector = torch.from_numpy(np.array(node_vector,dtype=np.int64)).to(device)
    det_J_t = torch.from_numpy(det_J_t).float().to(device)

    mua_bkg = torch.tensor(np.array(geom['bkg_mua'])).to(device)
    mus_bkg = torch.tensor(np.array(geom['bkg_mus'])).to(device)

    coords = torch.from_numpy(geom['coords']).float().to(device)
    elem = torch.from_numpy(np.array(geom['elem'],dtype=np.int64)).to(device)
    
    n = torch.tensor(np.array(n,dtype=np.int64)).to(device)
 
    Lmua = torch.from_numpy(geom['Lmua']).float().to(device)
    Lmus = torch.from_numpy(geom['Lmus']).float().to(device)

    if geom['solver'] == "GD":
        solver = 0
    elif geom['solver'] == "GN":
        solver = 1
    else:
        solver = 2
        
    qvec = geom['qvec']
    ns = geom['ns']
    qvec_torch = np.zeros((n,ns))
    for t in range(ns):
        qvec_torch[:,[t]] = qvec[n*t:n*(t+1)]
    
    qvec_torch = torch.from_numpy(qvec_torch).float().to(device)
    ns = torch.tensor(ns).to(device)
    log_data = geom['log_data']
    
    geom_torch = Qutil.geom_specs(B, mua_bkg, mus_bkg, 
                        n,coords,elem,V_fff_torch,
                        V_fdd_torch,Lmua,Lmus,JJ,det_J_t,node_matrix,node_vector,indice_matrix,log_scaling=log_data,solver=solver)
            

    data_torch = torch.from_numpy(dataSet.data).float().to(device)
    data_torch_test = torch.from_numpy(testSet.data).float().to(device)
      
    end = time.time()

    sTrain = len(dataSet.data[:,0])
    sTest = len(testSet.data[:,0])            
        
    current_input = torch.from_numpy(dataSet.initial).float().to(device)
    current_input_test = torch.from_numpy(testSet.initial).float().to(device)
    if log_data == False:
        Le_vec = torch.from_numpy(geom['Le_train']).float().to(device)
        Le_vec_test = torch.from_numpy(geom['Le_test']).float().to(device)
    else:
        Le_vec = 0
        Le_vec_test = 0

    if solver == 2:
            
            grad_prev = torch.from_numpy(prev_train.prev_grad).float().to(device)
            grad_prev_test = torch.from_numpy(prev_test.prev_grad).float().to(device)
            x_prev = torch.from_numpy(prev_train.prev_x).float().to(device)
            x_prev_test = torch.from_numpy(prev_test.prev_x).float().to(device)          
            x_prev = torch.flatten(torch.transpose(x_prev,3,4),3,4)
            x_prev = torch.cat((x_prev[:,:,0],x_prev[:,:,1]),2)         
            x_prev_test = torch.flatten(torch.transpose(x_prev_test,3,4),3,4)
            x_prev_test = torch.cat((x_prev_test[:,:,0],x_prev_test[:,:,1]),2)
         
            
    train_writer = tensorboardX.SummaryWriter(comment="/Greedy_"+str(lgsit)+"/train")
    test_writer = tensorboardX.SummaryWriter(comment="/Greedy_"+str(lgsit)+"/test")        


    with torch.no_grad():

        mua_all = torch.flatten(torch.transpose(current_input[:,[0]],2,3),2,3)
        mus_all = torch.flatten(torch.transpose(current_input[:,[1]],2,3),2,3)
    
        start = time.time()
    
        print("Computing gradients...")

        if solver == 2:
            dx_mua,dx_mus = Qutil.Get_grad_torch_greedy(geom_torch, mua_all, mus_all, data_torch, sTrain, ns, 
                                   Le_vec, qvec_torch, "cuda",cur_iter = currentLGSit,x_prev = x_prev,grad_prev = grad_prev)
        else:
            
            dx_mua,dx_mus = Qutil.Get_grad_torch_greedy(geom_torch, mua_all, mus_all, data_torch, sTrain, ns, 
                                   Le_vec, qvec_torch, "cuda")

        dx_mus_grid = torch.transpose(torch.reshape(dx_mus,(sTrain,1,ysize,xsize)),2,3)
        dx_mua_grid = torch.transpose(torch.reshape(dx_mua,(sTrain,1,ysize,xsize)),2,3) 
        current_dx_grid = torch.cat((dx_mua_grid,dx_mus_grid),dim=1)  

        print("and for test samples...")

        mua_all_test = torch.flatten(torch.transpose(current_input_test[:,[0]],2,3),2,3)
        mus_all_test = torch.flatten(torch.transpose(current_input_test[:,[1]],2,3),2,3)

        if solver == 2:
            dx_mua_test,dx_mus_test = Qutil.Get_grad_torch_greedy(geom_torch, mua_all_test, mus_all_test, data_torch_test, sTest, ns, 
                                           Le_vec_test, qvec_torch, "cuda",cur_iter = currentLGSit,x_prev = x_prev_test,grad_prev = grad_prev_test)                          
        else:
            dx_mua_test,dx_mus_test = Qutil.Get_grad_torch_greedy(geom_torch, mua_all_test, mus_all_test, data_torch_test, sTest, ns, 
                                    Le_vec_test, qvec_torch, "cuda")



        dx_mus_test_grid = torch.transpose(torch.reshape(dx_mus_test,(sTest,1,ysize,xsize)),2,3)
        dx_mua_test_grid = torch.transpose(torch.reshape(dx_mua_test,(sTest,1,ysize,xsize)),2,3)       
        current_dx_test_grid = torch.cat((dx_mua_test_grid,dx_mus_test_grid),dim=1)

        print("Done!")

        model = Nets.LGS(loss, xsize, ysize,n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lValInit)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter) 
        time_forw = 0
        time_back = 0
    
    for it in range(trainIter):
        
        start = time.time()
        batch = dataSet.next_batch(bSize)                

        batch_images = torch.from_numpy(batch[1]).float().to(device)
        batch_f = torch.from_numpy(batch[2]).float().to(device)            
        batch_indices = dataSet._perm[dataSet.start:dataSet.end]
        batch_grad = current_dx_grid[batch_indices]
        
        model.train()    
        optimizer.zero_grad(set_to_none=True)
        batch_output,loss = model(batch_f, batch_images,batch_grad)
        if (~torch.isfinite(batch_output)).any():
            print("Nan or Inf in output")
       
        end = time.time()
              
        time_forw += end-start
            
        # Take training step
        start = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()                  
        end = time.time()
        time_back += end-start

        if it % 2500 == 0 and it > 0:
            print('it: '+ str(it))
            print("Forward took (s): " +str(round(time_forw/500,5)))
            time_forw = 0  
            print("Training took (s): " + str(round(time_back/1001,5)))
            time_back = 0
            print("Peak cuda memory usage (GB): " + str(torch.cuda.max_memory_reserved(device=device)/10**9))
            
        
        # Evaluate loss and statistics
        if it % 2500 == 0:    
            if(useTensorboard):
                        
                with torch.no_grad():

                    model.eval()

                    if it == 0:

                        samples_test = len(testSet.data[:,0])
                        samples_train = len(dataSet.data[:,0])
                        max_sample = 50
                        max_sample_test = 50 
                        if samples_train < max_sample:
                            outputs_eval = torch.zeros(samples_train,2,xsize,ysize).float().to(device)
                            true_images_eval = torch.from_numpy(dataSet.true).float().to(device)
                            
                        else:
                            samples_train = max_sample
                            outputs_eval = torch.zeros(max_sample,2,xsize,ysize).float().to(device)
                            true_images_eval = torch.from_numpy(dataSet.true[0:max_sample]).float().to(device)
       

                        if samples_test < max_sample_test:
                            out_test_eval = torch.zeros(samples_test,2,xsize,ysize).float().to(device)
                            true_im_test_eval = torch.from_numpy(testSet.true).float().to(device)
                           
                        else:
                            samples_test = max_sample_test
                            out_test_eval = torch.zeros(samples_test,2,xsize,ysize).float().to(device)
                            true_im_test_eval = torch.from_numpy(testSet.true[0:samples_test]).float().to(device)
                                                         
                  
                    
                    loss_train = 0                   
                    
                    for ii in range(len(outputs_eval)):
                        single_f = current_input[[ii]]
                        single_true = true_images_eval[[ii]]
                        single_grad = current_dx_grid[[ii]]

                        outputs_eval[ii],loss = model(single_f,single_true,single_grad)
                        loss_train += loss

                    Qutil.summaries(train_writer,outputs_eval,true_images_eval,
                              loss_train, it,1)
                    
                    loss_test = 0
                    for ii in range(samples_test):
                        test_f = current_input_test[[ii]]
                        test_true = true_im_test_eval[[ii]]
                        test_grad = current_dx_test_grid[[ii]]                        

                        out_test_eval[ii],loss = model(test_f,test_true,test_grad)
                        loss_test += loss

                    
                    Qutil.summaries(test_writer,out_test_eval ,true_im_test_eval,
                              loss_test, it,1)
                
        
    print("Updating inputs for next networks")       

    model.eval()
      
    true_images = torch.from_numpy(dataSet.true).float().to(device)
    new_input = torch.zeros(sTrain,2,xsize,ysize).to(device)

    for ii in range(sTrain):
        single_f = current_input[[ii]]
        single_true = true_images[[ii]]
        single_grad = current_dx_grid[[ii]]

        new_input[ii],loss = model(single_f,single_true,single_grad)
        loss_train += loss
        
    true_im_test = torch.from_numpy(testSet.true).float().to(device)
    new_input_test = torch.zeros(sTest,2,xsize,ysize).to(device)                
    for ii in range(sTest):
        test_f = current_input_test[[ii]]
        test_true = true_im_test[[ii]]
        test_grad = current_dx_test_grid[[ii]]
    
        new_input_test[ii],loss = model(test_f,test_true,test_grad)
        loss_test += loss

    # Determine correct folder 
    run_nro = 0;
    
    while os.path.isdir("reconstructions/greedy_run_"+geom['solver']+"_"+str(run_nro)+"/"):
        run_nro +=1
    if lgsit > 0:
        run_nro -=1
    reco_folder = "reconstructions/greedy_run_"+geom['solver']+"_"+str(run_nro)+"/"  
    if lgsit == 0:
        os.mkdir(reco_folder)
    
    
    ##### Save reconstructions (.mat file) ##### 
    
    # Training samples
    out_train = new_input.detach().cpu().numpy()
    out_file_path = reco_folder+"train_" + experimentName + "greedy_model_"+ str(lgsit)+ '_outputs.mat'  
    scipy.io.savemat(out_file_path, {'trainout': out_train})
    
    # Test/validation samples
    out_numpy = new_input_test.detach().cpu().numpy()
    out_file_path = reco_folder+"test_" + experimentName + "greedy_model_"+ str(lgsit)+ '_outputs.mat'
    scipy.io.savemat(out_file_path, {'testout': out_numpy})
        
    ###### Save model parameters #####
    torch.save(model.state_dict(), filePath + experimentName + "_" + str(lgsit+1) + "_of_" + str(LGSiter))
        
    if solver == 2:
        return new_input.cpu().detach().numpy(), new_input_test.cpu().detach().numpy(),grad_prev[currentLGSit].cpu().detach().numpy(),grad_prev_test[currentLGSit].cpu().detach().numpy()      
    else:        
        return new_input.cpu().detach().numpy(), new_input_test.cpu().detach().numpy()

























