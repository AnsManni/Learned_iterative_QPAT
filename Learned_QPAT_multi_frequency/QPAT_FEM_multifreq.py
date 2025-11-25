
from scipy import sparse
import torch
import os
import numpy as np



def SysmatComponent(operation,geom,mua=[],kap=[]):

    elem = geom['elem']
    vtx = geom['coords']
    m = len(elem[:,0])
    # Boundary term
    n = geom['n']
    K = np.zeros((n,n))       

    if operation == 'BndPFF':
        gamma = 1/np.pi
        boundary_nodes = geom['bound_nodes']
        for t in range(len(boundary_nodes[:,0])):
            nodes = boundary_nodes[t,:];
            norm_F = np.linalg.norm(vtx[nodes[0],:]-vtx[nodes[1],:]);
            B_local = np.array([[1/3, 1/6] , [1/6, 1/3]])*norm_F*2*gamma;
            K[np.ix_(nodes,nodes)] +=  B_local;


    # Kappa term
    elif operation == 'PDD' or operation == 'PFF':
        
        M_base = np.zeros((3,3,3))

        for j in range(3):
            for jj in range(3):
                for jjj in range(3):
                    if j ==jj and jj == jjj:
                        M_base[j,jj,jjj] = 1/20
                    elif j == jj or jj==jjj or jjj==j:
                        M_base[j,jj,jjj] = 1/60
                    else:
                        M_base[j,jj,jjj] = 1/120


        for i in range(m):
        
            nodes =  (elem[i,:]).tolist()
            
            mua_cur = mua[nodes]

            x1 = vtx[nodes[0],:];
            x2 = vtx[nodes[1],:];
            x3 = vtx[nodes[2],:];
            
            # Jacobian of the transform
            X = np.squeeze(np.array([[x1],[x2],[x3]]))
            L = np.array([[-1, 1, 0] , [-1, 0, 1]])
            J_t = np.matmul(L,X)
            JJ = np.linalg.inv(J_t).dot(L)
            
            M_local = np.einsum('i,ijk',mua_cur,M_base) 
        
            K[np.ix_(nodes,nodes)] += (np.sum(kap[nodes]) * 1/6 * np.matmul(np.transpose(JJ),JJ)+M_local) * np.abs(np.linalg.det(J_t))

        K = sparse.csr_matrix(K)

    return K



def SysmatComponent_auxiliary(geom,to_pytorch=False,device='cpu'):

    elem = geom['elem']
    vtx = geom['coords']
    m = len(elem[:,0])
    # Boundary term
    JJ = np.zeros((m,3,3))
    det_J_t = np.zeros(m)
    n = geom['n']
    node_matrix = np.zeros((m,n))
    node_vector = np.zeros(3*m)

    indices_elem = np.zeros((m*9,2))


    for i in range(m):
    
        nodes =  (elem[i,:]).tolist()
        tp = 0
        for nod in nodes:
            node_matrix[i,nod] = 1
            node_vector[i*3+tp] = nod
            tp +=1
        
        
        t = 0
        tt = 0
        for p in range(9):
            indices_elem[p+i*9,:] = np.array([nodes[tt],nodes[t]])
            if (t+1) % 3:   
                t += 1
            else:
                t = 0
                tt += 1
        x1 = vtx[nodes[0],:];
        x2 = vtx[nodes[1],:];
        x3 = vtx[nodes[2],:];
        
        # Jacobian of the transform
        X = np.squeeze(np.array([[x1],[x2],[x3]]))
        L = np.array([[-1, 1, 0] , [-1, 0, 1]])
        J_t = np.matmul(L,X)
        JJ_temp = np.linalg.inv(J_t).dot(L)
        
        JJ[i,:,:] = 1/6 * np.matmul(np.transpose(JJ_temp),JJ_temp)
        det_J_t[i] = np.abs(np.linalg.det(J_t))
    
    indice_matrix_indices = []
    indice_matrix_values  = []
    for i in range(9*m):
        ind = n*indices_elem[i,0]+indices_elem[i,1]
        indice_matrix_indices.append([i,ind]) 
        indice_matrix_values.append(1)
        
    indice_matrix = torch.sparse_coo_tensor(list(zip(*indice_matrix_indices)),indice_matrix_values,size=(9*m,n**2))
    
    M_base = torch.zeros((3,3,3))
    
    for j in range(3):
        for jj in range(3):
            for jjj in range(3):
                if j ==jj and jj == jjj:
                    M_base[j,jj,jjj] = 1/20
                elif j == jj or jj==jjj or jjj==j:
                    M_base[j,jj,jjj] = 1/60
                else:
                    M_base[j,jj,jjj] = 1/120

    if to_pytorch == True:
        indice_matrix = indice_matrix.float().to(device)
        JJ = torch.from_numpy(JJ).float().to(device)
        node_matrix = torch.from_numpy(node_matrix).float().to_sparse().to(device)
        node_vector = torch.from_numpy(np.array(node_vector,dtype=np.int64)).to(device)
        det_J_t = torch.from_numpy(det_J_t).float().to(device)
        indices_elem = torch.from_numpy(np.array(indices_elem,dtype=np.int64)).to(device) 
        M_base = M_base.to(device)

    
    
        
    return JJ,det_J_t,indices_elem,node_matrix,node_vector,torch.transpose(indice_matrix,0,1),M_base



def SysmatComponent_torch(geom,mua,kap,device):

    elem = geom.elem
    m = len(elem[:,0])
    # Boundary term
    n = geom.n
    
    
    # Kappa term
    K = torch.zeros((n,n),device = torch.device(device))     
    '''
    M_base = torch.zeros((3,3,3),device = torch.device(device))
    
    for j in range(3):
        for jj in range(3):
            for jjj in range(3):
                if j ==jj and jj == jjj:
                    M_base[j,jj,jjj] = 1/20
                elif j == jj or jj==jjj or jjj==j:
                    M_base[j,jj,jjj] = 1/60
                else:
                    M_base[j,jj,jjj] = 1/120
    '''
    
    M_base = geom.M_base
    #M_base = geom.M_base

    sums =  torch.sparse.mm(geom.node_matrix,kap)
    adda = torch.mul(geom.JJ,sums[:,:,None])    
    

    mua_temp =  torch.gather(mua[:,0],0,geom.node_vector)
    mua_temp = torch.reshape(mua_temp,(m,3))
        
    M_local = torch.einsum('pi,ijk',mua_temp,M_base)  

    add = (adda + torch.permute(M_local,(2,1,0)))
    add = torch.mul(add,geom.det_J_t[:,None,None])

    
    # Make sparse matrix Q s.t. 
    # Q @ add = flatten(K) -> reshape to K
    add_flat = torch.flatten(add)

    K = torch.sparse.mm(geom.indice_matrix,add_flat[:,None])
    K = torch.reshape(K,(n,n))


    return K



def Generate_H_torch(geom,mua,mus,qvec,device):
    
    # Boundary term independent of mua, mus

    kap = 1/(2*(mua+mus))
    S1 = SysmatComponent_torch(geom, mua, kap,device) 

    K = S1+geom.B
    # Fluence
    phi = torch.linalg.solve(K,qvec) 

    H = torch.squeeze(torch.mul(mua,phi))

    return H,phi,K



def Get_grad_torch(geom, mua_batch, mus_batch, data,segments, bSize, 
                   qvec,C,stds,prior_vals,device):
    

    with torch.no_grad():
        
        if torch.is_tensor(bSize):
            bSize = bSize.item() 
    
        
        #n_in = geom.n_in
        freq_nro = 6
        freq_pairs = 3
        n_out = mua_batch.size()[-1] 
        dx_mua = torch.zeros((bSize,freq_pairs,n_out), device=torch.device(device))
        dx_mua_2 = torch.zeros((bSize,freq_pairs,n_out), device=torch.device(device))
        dx_mus = torch.zeros((bSize,freq_pairs,n_out), device=torch.device(device))
    
        for i in range(bSize):
    
            n_indices = torch.squeeze(torch.nonzero(segments[i,:]))
            n = n_indices.size(dim=0)
            
            J = torch.zeros((n,2*n), device=torch.device(device))
            
            if geom.solver > 0:
                
                Hes = torch.zeros(3*n,3*n,device=torch.device(device))
                grad_prior = torch.zeros(2*n,1,device=torch.device(device))
                diag_temp = torch.eye(n,n,device=torch.device(device))

            grad = torch.zeros(3*n,1,device=torch.device(device))
            
            if (i+1) % 50 == 0:
                print("Processing sample "+str(i+1)+" out of " +str(bSize))
                
            for f in range(0,freq_nro,2):
                
                qvec_scaled = torch.unsqueeze(qvec[:,f],1)
                if geom.solver > 0:
                    Lmua = 1/stds[0,i,f]
                    Lmus = 1/stds[1,i,f]
    
                mua_single = torch.unsqueeze(mua_batch[i,f],1)
                mus_single = torch.unsqueeze(mus_batch[i,f],1)
                    
    
                mua = mua_single[:,[0]]
                mus = mus_single[:,[0]]
                
                H,dphi,K = Generate_H_torch(geom, mua_single, mus_single, qvec_scaled, device) #takes very long...
                
                mua_circle = torch.index_select(mua,0,n_indices)
                mus_circle = torch.index_select(mus,0,n_indices)
                
                K_inv = torch.linalg.inv(K)  # takes long...
                coeff = 1/(2*torch.pow(mua+mus,2))
                
                A_temp = torch.sparse.mm(geom.V_fdd,dphi)
                A_temp_2 = torch.sparse.mm(geom.V_fff,dphi)    

                A_t = torch.transpose(torch.reshape(A_temp,(n_out,n_out)),0,1) 
                A_t = torch.mul(torch.transpose(coeff,0,1),torch.mm(K_inv,A_t))
                A_t = torch.index_select(A_t,1,n_indices)
                J[:,0:n] = torch.index_select(A_t,0,n_indices)
                J[:,n:2*n] =  J[:,0:n]

                # Derivative of the second term (mua only)
                
                A = torch.mm(K_inv,torch.transpose(torch.reshape(A_temp_2,(n_out,n_out)),0,1))
                A = torch.index_select(A,1,n_indices)
                J[:,0:n] = J[:,0:n] - torch.index_select(A,0,n_indices)
                    
                # times mua
                J = torch.mul(mua_circle.repeat(1,2*n),J)

                # Additional Jacobian term for mua         
                J[:,0:n] += torch.diag(torch.index_select(dphi[:,0],0,n_indices))

                #if positivity == "exp":
                #    J_1 = torch.mm(J_1,torch.diag(torch.cat((mua_circle[:,0],mus_circle[:,0]),0)))
                
                res = data[i,f]-torch.log(H)
                H_in = torch.index_select(H,0,n_indices)
                J_1 = torch.mm(torch.diag(1/H_in),J) 

                res = torch.unsqueeze(torch.index_select(res,0,n_indices),1)
                

                if geom.solver > 0: # solver == 0 is GD


                    J_prior_mua = Lmua
                    J_prior_mus = Lmus   
                        
                    priors_mua = torch.index_select(prior_vals[0,i,f],0,n_indices)
                    priors_mus = torch.index_select(prior_vals[1,i,f],0,n_indices)
    
                    grad_prior[0:n] =   torch.mul(Lmua*Lmua,mua_circle-priors_mua[:,None])
                    grad_prior[n:2*n] =  torch.mul(Lmus*Lmus,mus_circle-priors_mus[:,None])
    
                    grad_1 = torch.matmul(torch.transpose(J_1,0,1),res) -  grad_prior
                else:
                    
                    grad_1 = torch.matmul(torch.transpose(J_1,0,1),res)
                    
    
                if geom.solver > 0:  # 1 = Gauss-Newton
                    Hes_1 = J_1.T @ J_1
                    Hes_1[0:n,0:n] += J_prior_mua**2 * diag_temp
                    Hes_1[n:2*n,n:2*n] += J_prior_mus**2 * diag_temp

                q = f + 1
                for q in range(f+1,f+2):
    
                    if geom.solver == 1:
                        Hes[:,:] = 0
                    grad[:,:] = 0
                    if q != f:
    
                        qvec_scaled = torch.unsqueeze(qvec[:,q],1)
                        if geom.solver > 0:
                            Lmua = 1/stds[0,i,q]
                            Lmus = 1/stds[1,i,q]
        
                        mua_single = torch.unsqueeze(mua_batch[i,q],1)

                        mus_single_2 = torch.mul(mus_single,C[i,f,q])    
                        H,dphi,K = Generate_H_torch(geom, mua_single, mus_single_2, qvec_scaled, device)
    
                        mua = mua_single[:,[0]]
                        mus = mus_single_2[:,[0]]
                
                        mua_circle = torch.index_select(mua,0,n_indices)
                        mus_circle = torch.index_select(mus,0,n_indices)

                        K_inv = torch.linalg.inv(K)
    
                        # Form Jacobian 
                        # Derivative of the kap term 
                        coeff = 1/(2*torch.pow(mua+mus,2))
            
                        A_temp = torch.sparse.mm(geom.V_fdd,dphi)
                        A_temp_2 = torch.sparse.mm(geom.V_fff,dphi)

                        A_t = torch.transpose(torch.reshape(A_temp,(n_out,n_out)),0,1) 
                        A_t = torch.mul(torch.transpose(coeff,0,1),torch.mm(K_inv,A_t))
                        A_t = torch.index_select(A_t,1,n_indices)
                        
                        # second mua 
                        J[:,n:2*n] = torch.index_select(A_t,0,n_indices)
                        J[:,0:n] =  J[:,n:2*n]
                        
                        # Derivative of the second term (mua only)
                
                        A = torch.mm(K_inv,torch.transpose(torch.reshape(A_temp_2,(n_out,n_out)),0,1))
                        A = torch.index_select(A,1,n_indices)
                        J[:,n:2*n] = J[:,n:2*n] - torch.index_select(A,0,n_indices)
                         
                        # mua * J_fluence term
                        J = torch.mul(mua_circle.repeat(1,2*n),J)
                        
                        # Additional Jacobian term for mua         
                     
                        J[:,n:2*n] += torch.diag(torch.index_select(dphi[:,0],0,n_indices))
                
                        # Term from datascaling
                         
                        res = data[i,q]-torch.log(H)
                        
                        H_in = torch.index_select(H,0,n_indices)
                        J = torch.mm(torch.diag(1/H_in),J) 

                        #res = torch.transpose(res,0,1)
                        res = torch.unsqueeze(torch.index_select(res,0,n_indices),1)
    
                        if geom.solver > 0: # solver == 0 is GD
                

                            J_prior_mua = Lmua
                            J_prior_mus = Lmus   
                           
                            priors_mua = torch.index_select(prior_vals[0,i,q],0,n_indices)
                            priors_mus = torch.index_select(prior_vals[1,i,q],0,n_indices)
                            

                            grad_prior[n:2*n] =   torch.mul(Lmua*Lmua,mua_circle-priors_mua[:,None])
                            grad_prior[0:n] =  torch.mul(Lmus*Lmus,mus_circle-priors_mus[:,None])
                            
                            #grad_prior[n:2*n] =   J_prior_mua @ (Lmua @ (mua_circle-priors_mua[:,None]))
                            #grad_prior[0:n] = J_prior_mus @ (Lmus @ (mus_circle-priors_mus[:,None]))
                            
                            grad_2 = torch.matmul(torch.transpose(J,0,1),res) -  grad_prior
                        else:
                            grad_2 = torch.matmul(torch.transpose(J,0,1),res)
                        
                        grad[0:2*n,0] = grad_1[:,0]
                        grad[n:3*n,0] += grad_2[:,0]
    
                        # If using GN, form approximation of Hessian and compute direction
    
                        if geom.solver > 0:  # 1 = Gauss-Newton
                            Hes_2 = J.T @ J

                            Hes_2[0:n,0:n] += J_prior_mus**2 * diag_temp
                            Hes_2[n:2*n,n:2*n] += J_prior_mua**2 * diag_temp
    
                            Hes[0:2*n,0:2*n] = Hes_1
                            Hes[n:3*n,n:3*n] += Hes_2
    
                            dx = torch.linalg.solve(Hes,grad) 

                        else:
                            dx = grad
    
                        ind = int(f/2)
                        dx_mua[i,ind,n_indices] = dx[0:n,0]
                        dx_mus[i,ind,n_indices] = dx[n:2*n,0]
                        dx_mua_2[i,ind,n_indices] = dx[2*n:3*n,0]

    return dx_mua,dx_mus,dx_mua_2
    


def Get_grad_torch_EtoE(geom, cur_f,sample_info,device):
    

    mua_batch = cur_f[0]
    mua_batch_2 = cur_f[2]
    mus_batch = cur_f[1]
    qvec = sample_info.qvec
    stds = sample_info.stds
    data = sample_info.data
    
    
    if geom.solver > 0:
        prior_vals = sample_info.prior_vals
    C = sample_info.C
    
    bSize = 1
    n_out = mua_batch.size()[-1] 
    
    dx_mua = torch.zeros((bSize,n_out), device=torch.device(device))
    dx_mua_2 = torch.zeros((bSize,n_out), device=torch.device(device))
    dx_mus = torch.zeros((bSize,n_out), device=torch.device(device))

    for i in range(bSize):
        
        n_indices = torch.squeeze(torch.nonzero(sample_info.segments[i,:]))
        n = n_indices.size(dim=0)
        
        J_1 = torch.zeros((n,2*n), device=torch.device(device))
        J_2 = torch.zeros((n,2*n), device=torch.device(device))
        
        
        if geom.solver > 0:
            Hes = torch.zeros(3*n,3*n,device=torch.device(device))
            diag_temp = torch.eye(n,n,device=torch.device(device))
            
        grad = torch.zeros(3*n,1,device=torch.device(device))
        qvec_scaled = qvec[:,[0]]
        
        if geom.solver > 0:
            Lmua = 1/stds[0,i,0]
            Lmus = 1/stds[1,i,0]
        else:
            Lmua = 0
            Lmus = 0
            
        mua_single = mua_batch[:,None]
        mus_single = mus_batch[:,None]

       
        if geom.solver > 0:
            J_1,J_prior_mua,J_prior_mus,grad_1 = Form_Jacobian(geom,mua_single,mus_single,qvec_scaled,n_out,n_indices,device,data[i,0],Lmua,Lmus,prior_vals[:,i,0])
            Hes_1 = J_1.T @ J_1
            Hes_1[0:n,0:n] += J_prior_mua*J_prior_mua*diag_temp
            Hes_1[n:2*n,n:2*n] += J_prior_mus*J_prior_mus*diag_temp
        
        else:
            grad_1 = Form_Jacobian(geom,mua_single,mus_single,qvec_scaled,n_out,n_indices,device,data[i,0],Lmua,Lmus)

        # Second frequency
        qvec_scaled = qvec[:,[1]]
        if geom.solver > 0:
            Lmua_2 = 1/stds[0,i,1]
            Lmus_2 = 1/stds[1,i,1]
        else:
            Lmua_2 = 0
            Lmus_2 = 0
        
        mua_single_2 = mua_batch_2[:,None]
        mus_single_2 = torch.mul(mus_single,C[i])
        
        if geom.solver > 0:
            J_2,J_prior_mua_2,J_prior_mus_2,grad_2 = Form_Jacobian(geom,mua_single_2,mus_single_2,qvec_scaled,n_out,n_indices,device,data[i,1],Lmua_2,Lmus_2,prior_vals[:,i,1],reversed_order = 1)
            Hes_2 = J_2.T @ J_2
            Hes_2[0:n,0:n] += J_prior_mus_2**2 * diag_temp
            Hes_2[n:2*n,n:2*n] += J_prior_mua_2**2 * diag_temp
        
            Hes[0:2*n,0:2*n] = Hes_1
            Hes[n:3*n,n:3*n] += Hes_2        
        
        else:
            grad_2 = Form_Jacobian(geom,mua_single_2,mus_single_2,qvec_scaled,n_out,n_indices,device,data[i,1],Lmua_2,Lmus_2)

        grad[0:2*n,0] = grad_1[:,0]
        grad[n:3*n,0] += grad_2[:,0]
        
        # If using GN, form approximation of Hessian and compute direction
        if geom.solver > 0:  # 1 = Gauss-Newton  
            grad = torch.linalg.solve(Hes,grad) 

        dx_mua[i,n_indices] = grad[0:n,0]
        dx_mus[i,n_indices] = grad[n:2*n,0]
        dx_mua_2[i,n_indices] = grad[2*n:3*n,0]

    return dx_mua,dx_mus,dx_mua_2
    



def Form_Jacobian(geom,mua,mus,qvec_scaled,n_out,n_indices,device,data,Lmua,Lmus,prior_vals=0,bSize=1,reversed_order = 0):
    
    n = n_indices.size(dim=0)
    H,dphi,K = Generate_H_torch(geom, mua, mus, qvec_scaled, device)
    
    if geom.solver > 0:
        grad_prior = torch.zeros(2*n,1,device=torch.device(device))
        
    mua_circle = torch.index_select(mua,0,n_indices)
    mus_circle = torch.index_select(mus,0,n_indices)

    J = torch.zeros((n,2*n), device=torch.device(device))

    K_inv = torch.linalg.inv(K)

    coeff = 1/(2*torch.pow(mua+mus,2))
    
    A_temp = torch.sparse.mm(geom.V_fdd,dphi)
    
    A_t = torch.transpose(torch.reshape(A_temp,(n_out,n_out)),0,1) 
    A_t = torch.mul(torch.transpose(coeff,0,1),torch.mm(K_inv,A_t))
    A_t = torch.index_select(A_t,1,n_indices)
    J[:,0:n] = torch.index_select(A_t,0,n_indices)

    J[:,n:2*n] =  J[:,0:n]

    
    # Derivative of the second term (mua only)
    A = torch.sparse.mm(geom.V_fff,dphi)                  
    A = torch.mm(K_inv,torch.transpose(torch.reshape(A,(n_out,n_out)),0,1))
    A = torch.index_select(A,1,n_indices)
    if reversed_order == 1:
        J[:,n:2*n] = J[:,n:2*n] - torch.index_select(A,0,n_indices)
    else:
        J[:,0:n] = J[:,0:n] - torch.index_select(A,0,n_indices)


    J = torch.mul(mua_circle.repeat(1,2*n),J)
    
    # Additional Jacobian term for mua         
    if reversed_order == 1:
        J[:,n:2*n] += torch.diag(torch.index_select(dphi[:,0],0,n_indices))
    else:
        J[:,0:n] += torch.diag(torch.index_select(dphi[:,0],0,n_indices))
    

    # Residual term
    res = data-torch.log(H)
    H_in = torch.index_select(H,0,n_indices)
    J = torch.mm(torch.diag(1/H_in),J)      
    res = torch.unsqueeze(torch.index_select(res,0,n_indices),1)
    
    if geom.solver > 0: # solver == 0 is GD

        J_prior_mua = Lmua
        J_prior_mus = Lmus
        priors_mua = torch.index_select(prior_vals[0],0,n_indices)
        priors_mus = torch.index_select(prior_vals[1],0,n_indices)
        if reversed_order == 1:
            grad_prior[n:2*n] =   torch.mul(Lmua*Lmua,mua_circle-priors_mua[:,None])
            grad_prior[0:n] =  torch.mul(Lmus*Lmus,mus_circle-priors_mus[:,None])
        else:
            grad_prior[0:n] =   torch.mul(Lmua*Lmua,mua_circle-priors_mua[:,None])
            grad_prior[n:2*n] =  torch.mul(Lmus*Lmus,mus_circle-priors_mus[:,None])

        grad = torch.matmul(torch.transpose(J,0,1),res) -  grad_prior
    else:
        
        grad = torch.matmul(torch.transpose(J,0,1),res)
            
    
    if geom.solver > 0:
        return J,J_prior_mua,J_prior_mus,grad
    else:
        return grad




