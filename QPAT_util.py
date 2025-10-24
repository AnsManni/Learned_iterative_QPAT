import torch
import os
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def Generate_H_torch(geom,mua,mus,ns,qvec,device):
    
    
    n = geom.n
    # Boundary term independent of mua, mus
    H = torch.zeros((1,ns*n),device = torch.device(device))

    kap = 1/(2*(mua+mus))

    S1 = SysmatComponent_torch(geom, mua, kap,device) 

    phi = torch.zeros((n,ns),device = torch.device(device))
    K = S1+geom.B

    # Fluence
    phi = torch.linalg.solve(K,qvec) 
    
    A = torch.mul(mua.repeat(1,ns),phi)
    for j in range(ns):
        H[[0],j*n:(j+1)*n] = A[:,j]


    return H,phi,K


def SysmatComponent_torch(geom,mua,kap,device):

    elem = geom.elem
    m = len(elem[:,0])
    # Boundary term
    n = geom.n
    # Kappa term
    K = torch.zeros((n,n),device = torch.device(device))     
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


    sums =  torch.sparse.mm(geom.node_matrix,kap)
    adda = torch.mul(geom.JJ,sums[:,:,None])    
    

    mua_temp =  torch.gather(mua[:,0],0,geom.node_vector)#[nodes,0]
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



def Form_V(geom,file_fff,file_fdd):
          
    if os.path.isfile(file_fff):
        print("Loading sparse matrices")
        V_fff_torch = torch.load(file_fff)
        V_fdd_torch = torch.load(file_fdd)
        print("Completed")
        
    else:    
        print("Building FEM massMat")
        M_base = np.zeros((3,3,3))
        
        for j in range(3):
            for jj in range(3):
                for jjj in range(3):
                    if j ==jj and jj == jjj:
                        M_base[jj,jjj,j] = 1/20
                    elif j == jj or jj==jjj or jjj==j:
                        M_base[jj,jjj,j] = 1/60
                    else:
                        M_base[jj,jjj,j] = 1/120
        
        V_fff = []
        V_fdd = []
        
        n = geom['n']
        elem = geom['elem']-1
        vtx = geom['coords']
        L = np.array([[-1, 1, 0] , [-1, 0, 1]])
    
        for i in range(n):
            ind = np.where((elem==i))
            ind = ind[0]
            temp_fff = np.zeros((n,n));
            temp_fdd = np.zeros((n,n));
            for j in range(len(ind)):
                nodes = elem[ind[j],:];
                main_ind =  np.where((elem[ind[j],:]==i));
                main_ind =  main_ind[0]
                x1 = vtx[nodes[0],:];
                x2 = vtx[nodes[1],:];
                x3 = vtx[nodes[2],:];
                
                
                
                
                # Jacobian of the transform
                X = np.squeeze(np.array([[x1],[x2],[x3]]))
                J_t = np.matmul(L,X)
                JJ = np.linalg.inv(J_t).dot(L)
                add = 1/6 * np.matmul(np.transpose(JJ),JJ) * np.abs(np.linalg.det(J_t))
                temp_fdd[np.ix_(nodes,nodes)] += add;
    
                # Total V_fff is V_fdd + V_fff
                temp_fff[np.ix_(nodes,nodes)] += M_base[:,:,main_ind[0]] * np.abs(np.linalg.det(J_t))
    
    
                
            if i % 100 == 0:
                print('Nodes build: ',i)
            temp_fff = sparse.csr_matrix(temp_fff)
            temp_fdd = sparse.csr_matrix(temp_fdd)
            V_fff.append(temp_fff)
            V_fdd.append(temp_fdd)
    
            #sparse.save_npz(file_fff+"_" +str(i)+ ".npz",temp_fff)
            #sparse.save_npz(file_fdd+"_" +str(i)+ ".npz",temp_fdd)
            
        
        
        print("Converting to sparse matrices")
        V_fdd_torch = torch.from_numpy(V_fdd[0].todense())
        V_fff_torch = torch.from_numpy(V_fff[0].todense())
        V_fff_torch=V_fff_torch.to_sparse()
        V_fdd_torch=V_fdd_torch.to_sparse()  
        
        for mat in range(n-1):
            V_temp = torch.from_numpy(V_fdd[mat+1].todense())
            V_temp = V_temp.to_sparse()
            V_fdd_torch = torch.cat((V_fdd_torch,V_temp),dim=0)
    
            V_temp = torch.from_numpy(V_fff[mat+1].todense())
    
            V_temp = V_temp.to_sparse()
            V_fff_torch = torch.cat((V_fff_torch,V_temp),dim=0)
            if mat % 100 == 0:
                print(str(mat)+" out of " +str(n-1))
        torch.save(V_fff_torch, file_fff)
        torch.save(V_fdd_torch, file_fdd)


    return  V_fff_torch,V_fdd_torch

def Vector_to_grid(v,grid_coords,xsize,ysize,n):
    m = v.shape[0]
    u = np.zeros((m,2,xsize,ysize))

    for i in range(n):
        for j in range(m):
            u[j,0,grid_coords[i,0],grid_coords[i,1]] = v[j,0,i]
            u[j,1,grid_coords[i,0],grid_coords[i,1]] = v[j,1,i]

    
    return u


class DataSet(object):

  def __init__(self, data, true,initial):
    """Construct a DataSet"""
    
    self._num_examples = data.shape[0]

    self._data = data
    self._true = true
    self._initial = initial
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._start = 0
    self._end = 0
    self._temp_grad = np.zeros(true.shape)
    self._perm = np.arange(0,len(true[:,0,0]))
    
    
  @property
  def perm(self):
    return self._perm 
    
  @property
  def data(self):
    return self._data

  @property
  def temp_grad(self):
    return self._temp_grad

  @property
  def true(self):
    return self._true
 
  @property
  def initial(self):
    return self._initial


  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def start(self):
    return self._start

  @property
  def end(self):
    return self._end

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._perm = perm
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    self._start = start
    self._end = end
    return self._data[self._perm[start:end]], self._true[self._perm[start:end]], self._initial[self._perm[start:end]],self._temp_grad[self._perm[start:end]]


class Prev_values(object):

  def __init__(self,LGSiter,n,samples,xsize,ysize):
    """Construct a DataSet"""
    

    self._prev_x = np.zeros((LGSiter+1,samples,2,xsize,ysize))
    self._prev_grad = np.zeros((LGSiter,samples,2*n))

    
  @property
  def prev_grad(self):
    return self._prev_grad 
    
  @property
  def prev_x(self):
    return self._prev_x








def SysmatComponent_auxiliary(geom):

    elem = geom['elem']
    vtx = geom['coords']
    m = len(elem[:,0])
    # Boundary term
    JJ = np.zeros((m,3,3))
    det_J_t = np.zeros(m)
    n = geom['n']
    node_matrix = np.zeros((m,geom['n']))
    node_vector = np.zeros(3*m)

    indices_elem = np.zeros((m*9,2))


    for i in range(m):
    
        nodes =  (elem[i,:]-1).tolist()
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


    return JJ,det_J_t,node_matrix,node_vector,torch.transpose(indice_matrix,0,1)

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
            nodes = boundary_nodes[t,:]-1;
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
        
            nodes =  (elem[i,:]-1).tolist()
            
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

def Get_grad_torch_greedy(geom, mua_batch, mus_batch, data, bSize, ns, 
                   Le_vec, qvec, device,cur_iter = 0,x_prev=[],grad_prev=[]):
    

    with torch.no_grad():

        if torch.is_tensor(geom.n):
            n = geom.n.item() 
        else:
            n = geom.n
            
        if torch.is_tensor(ns):
            ns = ns.item() 
        
        if torch.is_tensor(bSize):
            bSize = bSize.item() 
        
        
        n = mua_batch.size()[-1]   
        
        dx_mua = torch.zeros((bSize,1,n), device=torch.device(device))
        dx_mus = torch.zeros((bSize,1,n), device=torch.device(device))
        J = torch.zeros((n*ns,2*n), device=torch.device(device))
        J_temp = torch.zeros((n*ns,n),device = torch.device(device))

        if geom.solver == 1:
            grad_prior = torch.zeros(2*n,1,device=torch.device(device))

        
        if geom.solver == 2:
            Id = torch.eye(2*n,device = torch.device(device))

        for i in range(bSize):
                       
            if (i+1) % 50 == 0 and i > 0:
                print("Computing step direction: " + str(i+1) + " out of " + str(bSize))
            
            if mua_batch[i].dim() == 2:
            
                mua_single = torch.transpose(mua_batch[i],0,1)
                mus_single = torch.transpose(mus_batch[i],0,1)
            
            else:           
                mua_single = torch.unsqueeze(mua_batch[i],1)
                mus_single = torch.unsqueeze(mus_batch[i],1)
            
            
            mua_single = torch.mul(torch.exp(mua_single),geom.bkg_mua)
            mus_single = torch.mul(torch.exp(mus_single),geom.bkg_mus)
                                  
            H,dphi,K = Generate_H_torch(geom, mua_single, mus_single, ns, qvec, device)

            mua = mua_single[:,[0]]
            mus = mus_single[:,[0]]
        
            
            K_inv = torch.linalg.inv(K)
            
            # Form Jacobian 
        
            # Derivative of the kap term 
        
            coeff = 1/(2*torch.pow(mua+mus,2))

            A_temp = torch.sparse.mm(geom.V_fdd,dphi)
        
            for jj in range(ns):
                A_t = torch.transpose(torch.reshape(A_temp[:,jj],(n,n)),0,1) 
        
                AA = torch.mm(K_inv,A_t)
                J_temp[jj*n:(jj+1)*n,:] = torch.mul(torch.transpose(coeff,0,1),AA)
                              
            J[:,0:n] = J_temp
            J[:,n:2*n] =  J[:,0:n]  
            
            # Derivative of the second term (mua only)
            A = torch.sparse.mm(geom.V_fff,dphi)
        
            for jj in range(ns):
                AA = torch.mm(K_inv,torch.transpose(torch.reshape(A[:,jj],(n,n)),0,1))
                J_temp[jj*n:(jj+1)*n,:] = AA
                       
            J[:,0:n] = J[:,0:n] - J_temp
            # mua * J_fluence term
            #J_clone = J#torch.clone(J)
        
        
            mua_temp = mua.repeat(ns,2*n)
            J = torch.mul(mua_temp,J)

            # Additional Jacobian term for mua         
         
            for jj in range(ns):
                J[jj*n:(jj+1)*n,0:n] += torch.diag(dphi[:,jj])
        
                          
            # If using scaled solution space
                
            #end = time.time()
            
            J = torch.mm(J,torch.diag(torch.cat((mua[:,0],mus[:,0]),0)))
        
            # Term from datascaling
            if geom.log_scaling == True:
                res = data[[i]]-torch.log(H)
                J = torch.mm(torch.diag(1/H[0,:]),J) 
                res = torch.transpose(res,0,1)

            else:
                res = data[[i]]-H
                Le = Le_vec[i,:]         
                res = torch.transpose(res,0,1)
                J = torch.multiply(Le[:,None],J) 
                res = torch.multiply(Le[:,None],res) 

            if geom.solver == 1: # solver == 0 is GD
        
                J_prior_mua = torch.mm(geom.Lmua,torch.diag(mua[:,0])) 
                J_prior_mus = torch.mm(geom.Lmus,torch.diag(mus[:,0])) 
                              
                grad_prior[0:n] =   J_prior_mua @ (geom.Lmua @ (mua-geom.bkg_mua))
                grad_prior[n:2*n] = J_prior_mus @ (geom.Lmus @ (mus-geom.bkg_mus))
                
                grad = torch.matmul(torch.transpose(J,0,1),res) -  grad_prior
            else:
                
                grad = torch.matmul(torch.transpose(J,0,1),res)
            if geom.solver == 0:
                dx = grad
                
            # If using GN, form approximation of Hessian and compute direction
        
            elif geom.solver == 1:  # 1 = Gauss-Newton
                Hes = J.T @ J
                Hes[0:n,0:n] += torch.matmul(torch.transpose(J_prior_mua,0,1),J_prior_mua)
                Hes[n:2*n,n:2*n] += torch.matmul(torch.transpose(J_prior_mus,0,1),J_prior_mus)
                dx = torch.linalg.solve(Hes,grad) 
        
            elif geom.solver == 2:   # SR1 method 
                r = 10**-8              
                grad_prev[cur_iter,i] = -grad[:,0]
                
                if cur_iter > 0:
                    #Hes_prev = Id   
                    x_prev[cur_iter,i] = torch.squeeze(torch.cat((torch.log(mua/geom.bkg_mua),torch.log(mus/geom.bkg_mus)),0))
                    
                    for lgs in range(cur_iter):                                                             
                        s = x_prev[lgs+1,i] - x_prev[lgs,i]
                        y = grad_prev[lgs+1,i] - grad_prev[lgs,i]
                        if lgs == 0:
                            Hes_prev = (torch.t(y) @ s) / (torch.t(y) @ y) * Id                                
                        V = (s-(Hes_prev @ y))
                        V = V[:,None]
                        nom = (V @ torch.t(V))
                        denom = (torch.t(V) @ y[:,None])
                        
                        a = torch.abs(denom)
                        b = torch.norm(y)*torch.norm(V)                               
                        
                        if a > r * b:
                            Hes_prev = Hes_prev + nom/denom
                        else:
                            print("Hessian wasn't updated")
                            Hes_prev = Hes_prev
                    
                    dx = Hes_prev @ grad    
                else:
                    dx = grad
                
        
        
            
            dx_mua[i,0,:] = dx[0:n,0]
            dx_mus[i,0,:] = dx[n:2*n,0]
        del dx, J, J_temp
        if geom.solver == 1:
            del Hes
        
        return dx_mua,dx_mus



def Get_grad_torch_EtoE(geom, mua_batch, mus_batch, data, bSize, ns, 
                   Le_vec, device, cur_iter = 0, Hes_prev=[],grad_prev=[],x_prev=[],B_prev=[],Id=[]):
    
    
    #start = time.time()
    if torch.is_tensor(geom.n):
        n = geom.n.item() 
    else:
        n = geom.n
        
    if torch.is_tensor(ns):
        ns = ns.item() 

    if torch.is_tensor(bSize):
        bSize = bSize.item() 


    n = mua_batch.size()[-1]   

    dx_mua = torch.zeros((bSize,1,n), device=torch.device(device))
    dx_mus = torch.zeros((bSize,1,n), device=torch.device(device))
    J = torch.zeros((n*ns,2*n), device=torch.device(device))
        
        
    for i in range(bSize):
        
        if mua_batch[i].dim() == 2:
        
            mua_single = torch.transpose(mua_batch[i],0,1)
            mus_single = torch.transpose(mus_batch[i],0,1)
        
        else:
        
            mua_single = torch.unsqueeze(mua_batch[i],1)
            mus_single = torch.unsqueeze(mus_batch[i],1)
        
        
        mua_single = torch.mul(torch.exp(mua_single),geom.bkg_mua)
        mus_single = torch.mul(torch.exp(mus_single),geom.bkg_mus)
        
                    
        H,dphi,K = Generate_H_torch(geom, mua_single, mus_single, ns, geom.qvec, device)
        
        mua = mua_single[:,[0]]
        mus = mus_single[:,[0]]

        
        K_inv = torch.linalg.inv(K)
              
        # Form Jacobian 

        # Derivative of the kap term 

        coeff = 1/(2*torch.pow(mua+mus,2))
        J_temp = torch.zeros((n*ns,n),device = torch.device(device))
        
        A_temp = torch.sparse.mm(geom.V_fdd,dphi)

        for jj in range(ns):
            A_t = torch.transpose(torch.reshape(A_temp[:,jj],(n,n)),0,1) 

            AA = torch.mm(K_inv,A_t)

            J_temp[jj*n:(jj+1)*n,:] = torch.mul(torch.transpose(coeff,0,1),AA)
               
        
        J[:,0:n] = J_temp
        J[:,n:2*n] =  J[:,0:n]
        
        
        # Derivative of the second term (mua only)
        A = torch.sparse.mm(geom.V_fff,dphi)

        for jj in range(ns):
            AA = torch.mm(K_inv,torch.transpose(torch.reshape(A[:,jj],(n,n)),0,1))
            J_temp[jj*n:(jj+1)*n,:] = AA

                
        J[:,0:n] = J[:,0:n] - J_temp

         
        # mua * J_fluence term
        J_clone = torch.clone(J)
       
        mua_temp = mua.repeat(ns,2*n)
        J = torch.mul(mua_temp,J_clone)
        

        # Additional Jacobian term for mua         
     
        for jj in range(ns):
            J[jj*n:(jj+1)*n,0:n] += torch.diag(dphi[:,jj])
            
            
        #end = time.time()
        
        J = torch.mm(J,torch.diag(torch.cat((mua[:,0],mus[:,0]),0)))

        # Term from datascaling
         
        res = data[[i]]-H

        res = torch.transpose(res,0,1)
        Le = Le_vec[i,:]

        J = torch.multiply(Le[:,None],J) 
        res = torch.multiply(Le[:,None],res) 
        if geom.solver > -1: # solver == 0 is GD

            J_prior_mua = torch.mm(geom.Lmua,torch.diag(mua[:,0])) 
            J_prior_mus = torch.mm(geom.Lmus,torch.diag(mus[:,0])) 
                          
            grad_prior = torch.zeros(2*n,1,device=torch.device(device))
            grad_prior[0:n] =   J_prior_mua @ (geom.Lmua @ (mua-geom.bkg_mua))
            grad_prior[n:2*n] = J_prior_mus @ (geom.Lmus @ (mus-geom.bkg_mus))
            
            grad = torch.matmul(torch.transpose(J,0,1),res) -  grad_prior
                
        else:
            
            grad = torch.matmul(torch.transpose(J,0,1),res)
                  

        # If using GN, form approximation of Hessian and compute direction

        # solver: 0 = GD, 1 = GN, 2 = BFGS, 3 = SR1 

        if geom.solver > 0:  # 1 = Gauss-Newton
            if geom.solver == 1:
                Hes_forw = J.T @ J

                Hes_forw[0:n,0:n] += torch.matmul(torch.transpose(J_prior_mua,0,1),J_prior_mua)
                Hes_forw[n:2*n,n:2*n] += torch.matmul(torch.transpose(J_prior_mus,0,1),J_prior_mus)

                dx =  torch.linalg.solve(Hes_forw,grad)

            elif (geom.solver == 2) and cur_iter > 0:   # SR1 method 
                r = 10**-8
                s = torch.cat((torch.log(mua/geom.bkg_mua),torch.log(mus/geom.bkg_mus)),0) - torch.unsqueeze(x_prev[i],1)
                y = -grad - grad_prev
                V = (s-Hes_prev @ y)
                with torch.no_grad():
                    a = torch.abs(torch.t(y) @ V)
                    b = torch.norm(y)*torch.norm(V)
                if a > r * b:
                    Hes = Hes_prev + (V @ torch.t(V))/(torch.t(V) @ y)
                else:
                    print("Hessian wasn't updated")
                    Hes = Hes_prev
    
                dx = Hes @ grad
            else:

                Hes = Id
                dx = grad

   
        if geom.solver == 0:
            dx_mua[i,0,:] = grad[0:n,0]
            dx_mus[i,0,:] = grad[n:2*n,0]
        else:
            dx_mua[i,0,:] = dx[0:n,0]
            dx_mus[i,0,:] = dx[n:2*n,0]
            
    if geom.solver < 2:
        return dx_mua,dx_mus
    elif geom.solver == 2:
        return dx_mua,dx_mus,Hes,-grad



def Visualize_samples(values,fig_name,set_name,scale,bkg_mua=0.01,bkg_mus=2):
    
    if len(values)>2:
        
        samples = 3
    
    else:
        samples = 1
    values = np.transpose(values,(0,1,3,2))

    fig,ax = plt.subplots(samples,2)
    
    
    for s in range(samples): 
        if scale == 1:
    
            plt.subplot(samples,2,2*s+1)
            plt.imshow(np.exp(values[s,0,:,:])*bkg_mua)
            plt.colorbar()
    
            plt.title(set_name + '(a)')
    
            plt.axis('off')
            plt.subplot(samples,2,2*s+2)
            plt.imshow(np.exp(values[s,1,:,:])*bkg_mus)
            plt.title(set_name+' (s)')
            plt.colorbar()
            plt.axis('off')
        else:
            plt.subplot(samples,2,2*s+1)
            plt.imshow(values[s,0,:,:])
            plt.colorbar()
    
            plt.title(set_name+' (a)')
    
            plt.axis('off')
            plt.subplot(samples,2,2*s+2)
            plt.imshow(values[s,1,:,:])
            plt.title(set_name+' (s)')
            plt.colorbar()
            plt.axis('off')


    fig.savefig(fig_name)
    
    
def summary_image_impl(writer, name, tensor, it):
    image = tensor[0, 0]
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')



def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)   


def summaries(writer, result, true, loss, it, scale_var):

    
    if scale_var == 0:
        #residual = result - true
        relative = torch.norm((result - true)) / torch.norm((true))
        rel_abs = torch.norm(((result[:,[0],:,:]) - (true[:,[0],:,:]))  ) / torch.norm(((true[:,[0],:,:]))  )
        rel_scat= torch.norm(((result[:,[1],:,:]) - (true[:,[1],:,:]))  ) / torch.norm(((true[:,[1],:,:]))  )
    
    elif scale_var == 1:
        #residual = torch.exp(result) - true
        relative = torch.norm((torch.exp(result) - true)  ) / torch.norm((true)  )
        rel_abs = torch.norm((torch.exp(result[:,[0],:,:]) - (true[:,[0],:,:]))  ) / torch.norm(((true[:,[0],:,:])))
        rel_scat= torch.norm((torch.exp(result[:,[1],:,:]) - (true[:,[1],:,:]))  ) / torch.norm(((true[:,[1],:,:])))
    

    writer.add_scalar('loss', loss, it)

    writer.add_scalar('relative', relative, it)

    writer.add_scalar('relative abs', rel_abs, it)
    writer.add_scalar('relative scat', rel_scat, it)

    summary_image(writer, 'Mua out', result[None,None,0,0,:,:], it)
    summary_image(writer, 'Mus out', result[None,None,0,1,:,:], it)

    summary_image(writer, 'True Mua', true[None,None,0,0,:,:], it)
    summary_image(writer, 'True Mus', true[None,None,0,1,:,:], it)



class geom_specs:
    def __init__(self,B,mua_bkg,mus_bkg,
                n,coords,elem,V_fff,V_fdd,Lmua,Lmus,JJ,det_J_t,node_matrix,node_vector,indice_matrix,qvec=0,log_scaling=False,solver=0):
        
        
        # Scalar values 
        self.solver        =     solver           

        self.n             =     n                 # number of unknowns
        self.bkg_mua       =     mua_bkg           # background/mean value of absorption 
        self.bkg_mus       =     mus_bkg           # -||- scattering

        self.node_vector   =     node_vector
        
        self.B             =     B                 # Constant part of sysmat

        self.V_fff         =     V_fff             # Sparse mass (integral) matrices
        self.V_fdd         =     V_fdd

        
        self.coords        =     coords            # Node coordinates
        self.elem          =     elem              # Element node indices
        self.node_matrix   =     node_matrix
        
        self.Lmua          =     Lmua              # Prior Cholesky matrices
        self.Lmus          =     Lmus
        
        # Auxiliary variables for FEM system matrix computation
        self.det_J_t       =     det_J_t
        self.JJ            =     JJ
        self.indice_matrix =     indice_matrix
        self.qvec          =     qvec
        self.log_scaling   =     log_scaling

        
        
        
        
class prev_values:
    def __init__(self,H,x,grad):
        
        self.prev_H            = H             # Grid size
        self.prev_grad         = grad
        self.prev_x            = x     
        
        
        
        
        
def Interpolate_optical(data,mua,mus,geom,orig_coords,setname):            
            
    if os.path.isfile(setname):
        intp_dataset = torch.load(setname) 
        data_intp = torch.cat((intp_dataset[0],intp_dataset[1]),1).numpy()
        images = torch.cat((intp_dataset[[1]],intp_dataset[[2]]),0).swapaxes(0,1).numpy()


    else:   
        coords = torch.from_numpy(orig_coords).float()
        coords_intp = torch.from_numpy(geom["coords"]).float()
        mua = torch.from_numpy(mua)
        mus = torch.from_numpy(mus)
        data = torch.from_numpy(data)
        
        n = geom["n"]
        
        dh_init = 1/3
        
        weights = []
        indices = []
        min_ind = []
        for j in range(n):            
            dst = torch.norm(coords - coords_intp[j,:],dim=1)
            ind_neigh = torch.nonzero(torch.where(dst<1.8*dh_init,1,0)).squeeze()
            neigh_values = torch.index_select(dst, 0, ind_neigh)
            min_ind_neigh = torch.argmin(neigh_values)
            min_ind.append(ind_neigh[min_ind_neigh])
            indices.append(ind_neigh)
            weights.append(torch.exp(-8*neigh_values/(1.8*dh_init)))
    

        # freqs frequencies
        samples = len(mua)
        data_intp = torch.zeros((samples,2*n))
        mua_intp = torch.zeros((samples,n))
        mus_intp = torch.zeros((samples,n))
    
        for i in range(samples):
                
                if i % 50 == 0:
                    print("Sample "+str(i+1)+" being interpolated")
                 
                for t in range(n):
                    data_cur = torch.index_select(data[i],0,indices[t])
                    
                    mus_cur = torch.index_select(mus[i],0,indices[t])
                    mua_cur = torch.index_select(mua[i],0,indices[t])
                    
                    w_sum = torch.sum(weights[t])
                    # The coeffs are also scaled from 1/cm to 1/mm
                    mus_intp[i,t] = torch.sum(torch.multiply(weights[t],mus_cur))/w_sum
                    mua_intp[i,t] = torch.sum(torch.multiply(weights[t],mua_cur))/w_sum  
                    data_intp[i,t] = torch.sum(torch.multiply(weights[t],data_cur))/w_sum
                    data_intp[i,t+n] = torch.sum(torch.multiply(weights[t],data_cur))/w_sum
               
  
        intp_dataset = torch.zeros((4,samples,n))
        
        intp_dataset[0] = data_intp[:,0:n];
        intp_dataset[1] = data_intp[:,n:2*n];

        intp_dataset[2] = mua_intp
        intp_dataset[3] = mus_intp;
        # save datasets
        
        
        torch.save(intp_dataset,setname) 

        data_intp = torch.cat((intp_dataset[0],intp_dataset[1]),1).numpy()
        images = torch.cat((intp_dataset[[2]],intp_dataset[[3]]),0).swapaxes(0,1).numpy()
    
    
    return data_intp,images