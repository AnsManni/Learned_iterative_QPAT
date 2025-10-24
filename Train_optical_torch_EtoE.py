''' Class file using pytorch Learned image recontruction for PAT
https://doi.org/10.1117/1.JBO.25.11.112903
Written 2020 by Andreas Hauptmann, University of Oulu and UCL'''

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


def training_EtoE(dataSet,testSet,geom,netType,experimentName,filePath,
             lossFunc = 'l2_loss',
             bSize = 4,
             trainIter = 50001,
             LGSiter = 10,
             useTensorboard = True,
             lValInit=1e-3,
             device='cuda'):
    
    

    # Image dimensions

    xsize = geom['xsize']
    ysize =  geom['ysize']
    
    n = geom['n']
    # Constant part(s) of System matrix (Boundary part)
    B = torch.from_numpy(Qutil.SysmatComponent('BndPFF', geom)).float().to(device)

    loss = nn.MSELoss()
    print('Using end-to-end LGS: ' + str(LGSiter) + ' iterations')
    
    if(useTensorboard):
        train_writer = tensorboardX.SummaryWriter(comment="/LGS_EtoE_"+str(LGSiter)+"_"+geom['solver']+"/train")
        test_writer = tensorboardX.SummaryWriter(comment="/LGS_EtoE/test")
    
                
    print("Initializing the model inputs...")

    V_fdd = geom['V_fdd']
    V_fff = geom['V_fff']          
    JJ,det_J_t,node_matrix,node_vector,indice_matrix =  Qutil.SysmatComponent_auxiliary(geom)
    
    print("Transfering to Device")        
        
    mua_bkg = torch.tensor(np.array(geom['bkg_mua'])).to(device)
    mus_bkg = torch.tensor(np.array(geom['bkg_mus'])).to(device)

    coords = torch.from_numpy(geom['coords']).float().to(device)
    elem = torch.from_numpy(np.array(geom['elem'],dtype=np.int64)).to(device)
    n = torch.tensor(np.array(n,dtype=np.int64)).to(device)

    V_fdd = V_fdd.float().to(device)
    V_fff = V_fff.float().to(device)

    Lmua = torch.from_numpy(geom['Lmua']).float().to(device)
    Lmus = torch.from_numpy(geom['Lmus']).float().to(device)
    if geom['solver'] == "GD":
        solver = 0
    elif geom['solver'] == "GN":
        solver = 1
    elif geom['solver'] == "SR1":
        solver = 2

    log_data = geom['log_data']    
    indice_matrix = indice_matrix.float().to(device)
    JJ = torch.from_numpy(JJ).float().to(device)
    node_matrix = torch.from_numpy(node_matrix).float().to_sparse().to(device)
    node_vector = torch.from_numpy(np.array(node_vector,dtype=np.int64)).to(device)
    det_J_t = torch.from_numpy(det_J_t).float().to(device)
    model = Nets.LGS_EtoE(LGSiter,loss,n,xsize,ysize).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lValInit)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter)  
          
    qvec = geom['qvec']
    ns = geom['ns']
    qvec_torch = np.zeros((n,ns))
    for t in range(ns):
        qvec_torch[:,[t]] = qvec[n*t:n*(t+1)]
    
    qvec_torch = torch.from_numpy(qvec_torch).float().to(device)
    ns = torch.tensor(ns).to(device)
    bSize_torch = torch.tensor(bSize).to(device)
    
    time_forw = 0
    time_back = 0
    
    init_mua = torch.from_numpy(dataSet.initial[:,[0]]).float().to(device)
    init_mus = torch.from_numpy(dataSet.initial[:,[1]]).float().to(device)
    init_mua = torch.flatten(torch.transpose(init_mua,2,3),2,3)
    init_mus = torch.flatten(torch.transpose(init_mus,2,3),2,3)

    init_mua_test = torch.from_numpy(testSet.initial[:,[0]]).float().to(device)
    init_mus_test = torch.from_numpy(testSet.initial[:,[1]]).float().to(device)
    init_mua_test = torch.flatten(torch.transpose(init_mua_test,2,3),2,3)
    init_mus_test = torch.flatten(torch.transpose(init_mus_test,2,3),2,3)
    
    sTest = len(init_mua_test)
    sTrain = len(init_mua)
    
    data_train = torch.from_numpy(dataSet.data).float().to(device)
    data_test = torch.from_numpy(testSet.data).float().to(device)
    if log_data == False:
        Le_vec = torch.from_numpy(geom['Le_train']).float().to(device)
        Le_vec_test = torch.from_numpy(geom['Le_test']).float().to(device)
    else:
        Le_vec = 0
        Le_vec_test = 0
        
    geom_torch = Qutil.geom_specs(B, mua_bkg, mus_bkg, n, coords, elem, V_fff, V_fdd,
                                  Lmua, Lmus, JJ, det_J_t, node_matrix, node_vector, indice_matrix,qvec_torch,solver,log_data)
    
    print("Computing initial step directions...")
    
    dx_init_mua,dx_init_mus = Qutil.Get_grad_torch_EtoE(geom_torch,init_mua, init_mus, data_train, sTrain, ns, Le_vec,device)
    dx_init_mua_test,dx_init_mus_test = Qutil.Get_grad_torch_EtoE(geom_torch,init_mua_test, init_mus_test, data_test, sTest, ns, Le_vec_test,device)
    init_grad = torch.cat((dx_init_mua,dx_init_mus),dim=1)
    init_grad_test = torch.cat((dx_init_mua_test,dx_init_mus_test),dim=1)

    print("Done!")
    
    
    for it in range(trainIter):
        
        # feed_train={data: batch[0], true: batch[1], reco: batch[2], learningRate: lVal}
        batch = dataSet.next_batch(bSize)                

        batch_data = torch.from_numpy(batch[0]).float().to(device)

        batch_images = torch.from_numpy(batch[1]).float().to(device)
        batch_f = torch.from_numpy(batch[2]).float().to(device)            
        batch_indices = dataSet._perm[dataSet.start:dataSet.end]
        
        batch_init_grad = init_grad[batch_indices].float().to(device)
        if log_data == False:
            batch_Le = torch.from_numpy(geom['Le_train'][batch_indices,:]).float().to(device)
        else:
            batch_Le = 0
            
        model.train()    
        optimizer.zero_grad(set_to_none=True)
        
        start = time.time()
        batch_output,loss = model(batch_f,batch_data,batch_images,geom_torch,bSize_torch,ns,batch_Le,batch_init_grad,device)
        end = time.time()

        time_forw += end-start

        if it % 1000 == 0 and it > 0:
            print("Forward took:")
            print(time_forw/1001)
            time_forw = 0   

        
        start = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()   

        end = time.time()
        time_back += end-start


        if (~torch.isfinite(batch_output)).any():
            print("Nan or Inf in output")
    
        if it % 1000 == 0 and it > 0:
            print('it: '+ str(it))
            print("Training took: " + str(time_back/1001))
            time_back = 0
            print("RAM used (GB): " + str(torch.cuda.max_memory_allocated(device=device)/10**9))
            
            
        if it % 5000 == 0:    
            if(useTensorboard):
            
                with torch.no_grad():    
                    model.eval()
                    if it == 0:
                        samples_train = 20
                        samples_test = 20

                        outputs = torch.zeros(samples_train,2,xsize,ysize).to(device)
                        init_f = torch.from_numpy(dataSet.initial[0:samples_train]).float().to(device)
                        true_images = torch.from_numpy(dataSet.true[0:samples_train]).float().to(device)
                        data_train = torch.from_numpy(dataSet.data[0:samples_train]).float().to(device)
                        Le_train = torch.from_numpy(geom['Le_train'][0:samples_train]).float().to(device)
                               
                        if geom['test'] == True:                           
                            out_test = torch.zeros(samples_test,2,xsize,ysize).to(device)
                            init_f_test = torch.from_numpy(testSet.initial[0:samples_train]).float().to(device)
                            true_im_test = torch.from_numpy(testSet.true[0:samples_train]).float().to(device)
                            data_test = torch.from_numpy(testSet.data[0:samples_train]).float().to(device)
                            Le_test = torch.from_numpy(geom['Le_test'][0:samples_train]).float().to(device)
                                 
                   
                    loss_total = 0
                    for ii in range(samples_train):
                        single_data = data_train[[ii]]
                        single_f = init_f[[ii]]
                        single_true = true_images[[ii]]
                        single_Le = Le_train[[ii]]
                        single_init_grad = init_grad[[ii]]
                        
                        outputs[ii],loss = model(single_f,single_data,single_true,geom_torch,bSize_torch,
                                                ns,single_Le,single_init_grad,device)
                        loss_total+=loss
                    
                    b_out = outputs.clone()
                    b_im = true_images.clone()
                    
                    Qutil.summaries(train_writer,b_out,b_im,
                              loss_total, it, 0)
                    if geom['test'] == True:
                        
                        for ii in range(samples_test):
                            test_data = data_test[[ii]]
                            test_f = init_f_test[[ii]]
                            test_true = true_im_test[[ii]]
                            test_Le = Le_test[[ii]]
                            test_init_grad = init_grad_test[[ii]]

                            
                            out_test[ii],loss = model(test_f,test_data,test_true,geom_torch,bSize_torch,
                                                    ns,test_Le,test_init_grad,device)
                        
                        
                        out_t = out_test.clone()
                        test_im = true_im_test.clone()
                        
                        Qutil.summaries(test_writer,out_t ,test_im,
                                  loss, it, 0)
                
                
                    
    with torch.no_grad():  
        
        print("Computing final reconstructions...")
        model.eval()
        samples_train = len(dataSet.data[:,0])
        samples_test = len(testSet.data[:,0])
        output_recos = torch.zeros(samples_train,2,xsize,ysize).to(device)
        init_f = torch.from_numpy(dataSet.initial).float().to(device)        
        true_images = torch.from_numpy(dataSet.true).float().to(device)
        data_train = torch.from_numpy(dataSet.data).float().to(device)
        Le_train = torch.from_numpy(geom['Le_train']).float().to(device)
        
        
        for ii in range(samples_train):
            single_data = data_train[[ii]]
            single_f = init_f[[ii]]
            single_true = true_images[[ii]]
            single_Le = Le_train[[ii]]
            single_init_grad = init_grad[[ii]]

            output_recos[ii],loss = model(single_f,single_data,single_true,geom_torch,bSize_torch,
                                    ns,single_Le,single_init_grad,device)
          
        out_test = torch.zeros(samples_test,2,xsize,ysize).to(device) 
        init_f_test = torch.from_numpy(testSet.initial).float().to(device)
        true_im_test = torch.from_numpy(testSet.true).float().to(device)
        data_test = torch.from_numpy(testSet.data).float().to(device)
        Le_test = torch.from_numpy(geom['Le_test']).float().to(device)
             
        for ii in range(samples_test):
            test_data = data_test[[ii]]
            test_f = init_f_test[[ii]]
            test_true = true_im_test[[ii]]
            test_Le = Le_test[[ii]]
            test_init_grad = init_grad_test[[ii]]
            
            out_test[ii],loss = model(test_f,test_data,test_true,geom_torch,bSize_torch,
                                    ns,test_Le,test_init_grad,device)
    
    
    print("Done!")

    
    torch.cuda.synchronize(device=device)   
    
    out_reco_test = out_test.detach().cpu().numpy()
    out_reco_train = output_recos.detach().cpu().numpy()

    file_path_test ="reconstructions/EtoE_"+geom['solver']+"/test" + experimentName + '_outputs.mat'
    file_path_train ="reconstructions/EtoE_"+geom['solver']+"/train" + experimentName + '_outputs.mat'
    
    if ~os.path.isdir("reconstructions/EtoE_"+geom['solver']+"/"):
        os.mkdir("reconstructions/EtoE_"+geom['solver']+"/")
             
    scipy.io.savemat(file_path_train, {'trainout': out_reco_train})
    scipy.io.savemat(file_path_test, {'testout': out_reco_test})
    
    torch.save(model.state_dict(), filePath + experimentName)
    
    return

            
