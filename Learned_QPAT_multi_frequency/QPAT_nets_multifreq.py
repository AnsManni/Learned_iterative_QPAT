import torch
from torch import nn
import QPAT_FEM_multifreq as FE



class LGS(nn.Module):
       
       def __init__(self, loss,xsize,ysize,cur_it,niter=1,B=[],
                    scaled_data=[],n=[],grid_coords=[],coords=[],elem=[],V_fff=[],V_fdd=[],solver=[],JJ=[],det_J_t=[],
                    indices_elem=[],node_matrix=[],node_vector=[],indice_matrix=[],greedy=1,combined=False,cropped=False,data_assisted=False):
           super().__init__()
           

           # Scalar values 
           self.xsize         =     xsize             # Grid size
           self.ysize         =     ysize
           self.n             =     n                 # number of unknowns

           self.solver        =     solver            # GN or GD
           self.scaled_data   =     scaled_data       # Scaled data space
           self.niter         =     niter             # LGS iterations
           self.node_vector   =     node_vector
           self.greedy        =     greedy
           
           self.B             =     B                 # Constant part of sysmat
           self.loss          =     loss              # Loss functions

           self.V_fff         =     V_fff             # Sparse mass (integral) matrices
           self.V_fdd         =     V_fdd

           
           #self.coords        =     coords            # Node coordinates
           self.elem          =     elem              # Element node indices
           self.node_matrix   =     node_matrix

           
           # Auxiliary variables for FEM system matrix computation
           self.det_J_t       =     det_J_t
           self.JJ            =     JJ
           self.indices_elem  =     indices_elem
           self.indice_matrix =     indice_matrix
           self.combined      =     combined
           self.cropped       =     cropped
           self.cur_it        =     cur_it
           self.data_assisted =     data_assisted
           
           
           
           if combined == True:
               mm = 6
               c_out = 3
           else:
               if data_assisted == True:
                   mm = 3
               else:
                   mm = 2
                   
               c_out = 1 
               
               
           for i in range(niter):
               iteration_mua = Iteration(mm,c_out)
               setattr(self, 'iteration_mua_{}'.format(i), iteration_mua)
               if combined == "partially" and cur_it > 0:
                   iteration_mus = Iteration(mm+1,c_out)
               else:
                   iteration_mus = Iteration(mm,c_out)

               setattr(self, 'iteration_mus_{}'.format(i), iteration_mus)
           
           
           
       def forward(self,cur_f_whole,grad_f,segments,true,mus_weight,C,data=[],qvec=[],bSize=1,data_means=[],cur_segments=[],device='cpu',cropped_indices=0,cur_data=0):      

         
            cur_mua_vec = cur_f_whole[[0],:]
            cur_mus_vec = cur_f_whole[[1],:]
            cur_mua_vec_2 = cur_f_whole[[2],:]
            
            add_mua = torch.zeros(1,1,cur_mua_vec.size(dim=1),device = device)
            add_mua_2 = torch.zeros(1,1,cur_mua_vec.size(dim=1),device = device)
            add_mus = torch.zeros(1,1,cur_mua_vec.size(dim=1),device = device) 
               
            
               
            if self.cropped == True:
                grad_vec = torch.zeros(1,1,self.ysize*self.xsize,device = device)
                grad_vec_2 = torch.zeros(1,1,self.ysize*self.xsize,device = device)
                grad_vec_3 = torch.zeros(1,1,self.ysize*self.xsize,device = device)

                cur_mua = torch.zeros(1,1,self.ysize*self.xsize,device = device)
                cur_mus = torch.zeros(1,1,self.ysize*self.xsize,device = device)
                cur_mua_2 = torch.zeros(1,1,self.ysize*self.xsize,device = device)
                cur_mua[0,0,cropped_indices[:,0]] =  cur_mua_vec
                cur_mua_2[0,0,cropped_indices[:,0]] =  cur_mua_vec_2
                cur_mus[0,0,cropped_indices[:,0]] =  cur_mus_vec
                
                
                grad_vec[0,0,cropped_indices[:,0]] = grad_f[0]
                grad_mua = grad_vec
                
                grad_vec_2[0,0,cropped_indices[:,0]] = grad_f[1]
                grad_mus = grad_vec_2
  
                grad_vec_3[0,0,cropped_indices[:,0]] = grad_f[2]
                grad_mua_2 = grad_vec_3 
                
            else:  
                cur_mua = torch.mul(cur_mua_vec,1)
                cur_mua_2 = torch.mul(cur_mua_vec_2,1)
                cur_mus = torch.mul(cur_mus_vec,1)
                grad_mua = grad_f[0]
                grad_mus = grad_f[1]             
                grad_mua_2 = grad_f[2]
                
            cur_mua = torch.transpose(torch.reshape(cur_mua[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)
            cur_mua_2 = torch.transpose(torch.reshape(cur_mua_2[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)
            cur_mus = torch.transpose(torch.reshape(cur_mus[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)
                     
            grad_mus = torch.transpose(torch.reshape(grad_mus[None,:],(self.ysize,self.xsize)),0,1)
            grad_mua = torch.transpose(torch.reshape(grad_mua[None,:],(self.ysize,self.xsize)),0,1) 
            grad_mua_2 = torch.transpose(torch.reshape(grad_mua_2[None,:],(self.ysize,self.xsize)),0,1) 
            
            iteration_mua = getattr(self, 'iteration_mua_{}'.format(0))
            iteration_mus = getattr(self, 'iteration_mus_{}'.format(0))
            segments_nro = torch.squeeze(torch.nonzero(segments))

            if self.combined == True:
            
                cur_all = torch.cat([cur_mua,cur_mua_2,cur_mus], dim=1)
                grad_all=torch.cat([grad_mua[None,None,:,:],grad_mua_2[None,None,:,:],grad_mus[None,None,:,:]], dim=1)

                inp_all = iteration_mua(cur_all,grad_all)
                inp_all = torch.flatten(torch.transpose(inp_all,2,3),2,3)
                inp_all = torch.index_select(inp_all,2,segments_nro)
                add_mua[0,0,segments_nro] = inp_all[:,[0]]
                add_mua_2[0,0,segments_nro] = inp_all[:,[1]]
                add_mus[0,0,segments_nro] = inp_all[:,[2]]    
            
            
            else:

                   
                if self.data_assisted == True:
                    inp_mua = iteration_mua(cur_mua,torch.cat([grad_mua[None,None,:,:],cur_data[:,[0]]],dim=1))
                    inp_mua_2 = iteration_mua(cur_mua_2,torch.cat([grad_mua_2[None,None,:,:],cur_data[:,[1]]],dim=1))
                else:
                    inp_mua = iteration_mua(cur_mua,grad_mua[None,None,:,:])
                    inp_mua_2 = iteration_mua(cur_mua_2,grad_mua_2[None,None,:,:])
                

                
                if self.combined =="partially" and self.cur_it > 0:
                    if self.data_assisted == True:
                        inp_mus = iteration_mus(torch.cat([cur_mus,cur_mua], dim=1),torch.cat([grad_mus[None,None,:,:],cur_data[:,[0]]],dim=1))
                    else:
                        inp_mus = iteration_mus(torch.cat([cur_mus,cur_mua], dim=1),grad_mus[None,None,:,:])

                        
                else:
                    if self.data_assisted == True:
                        inp_mus = iteration_mus(cur_mus,torch.cat([grad_mus[None,None,:,:],cur_data[:,[0]]],dim=1))

                    else:
                        inp_mus = iteration_mus(cur_mus,grad_mus[None,None,:,:])

            # To vectors
             
            inp_mus = torch.flatten(torch.transpose(inp_mus,2,3),2,3)
            inp_mua = torch.flatten(torch.transpose(inp_mua,2,3),2,3)
            inp_mua_2 = torch.flatten(torch.transpose(inp_mua_2,2,3),2,3)
            
             
            # Circle gradients 
            
            if self.cropped == True:
                inp_mua = torch.index_select(inp_mua,2,cropped_indices[:,0])
                inp_mua_2 = torch.index_select(inp_mua_2,2,cropped_indices[:,0])
                inp_mus = torch.index_select(inp_mus,2,cropped_indices[:,0])
            
            inp_mua = torch.index_select(inp_mua,2,segments_nro)
            inp_mua_2 = torch.index_select(inp_mua_2,2,segments_nro)
            inp_mus = torch.index_select(inp_mus,2,segments_nro)
             
            
            
            add_mua[0,0,segments_nro] = 0.01*inp_mua
            add_mua_2[0,0,segments_nro] = 0.01*inp_mua_2
            add_mus[0,0,segments_nro] = 0.01*inp_mus
            

               
            cur_mua_vec += add_mua[0]
            cur_mus_vec += add_mus[0]
            cur_mua_vec_2 += add_mua_2[0]
            
            #if self.positivity == "exp":
            #    cur_mus_vec_2 = torch.log((torch.exp(cur_mus_vec)*C))
            #else:
            cur_mus_vec_2 = cur_mus_vec*C
            
                
            cur_f = torch.cat((cur_mua_vec,cur_mus_vec,cur_mua_vec_2,cur_mus_vec_2),dim=0)
            
            #if self.positivity != "exp":

            a = 0.1 #torch.log(torch.tensor(0.05/mus_weight)).to(device)
            b = 0.001 #torch.log(torch.tensor(0.001)).to(device)
            a_max = 3     
            b_max = 1.2 
     
            cur_f_inner = torch.index_select(cur_f,1,segments_nro)
            cur_f_inner_temp = torch.zeros(cur_f_inner.size(dim=0),cur_f_inner.size(dim=1),device = device)
            cur_f_inner_temp[0] = torch.clamp(cur_f_inner[0], min=b,max=b_max)
            cur_f_inner_temp[1] = torch.clamp(cur_f_inner[1], min=a,max=a_max)/mus_weight
            cur_f_inner_temp[2] = torch.clamp(cur_f_inner[2], min=b,max=b_max)
            cur_f_inner_temp[3] = torch.clamp(cur_f_inner[3], min=a,max=a_max)/mus_weight

            cur_f[:,segments_nro] = torch.clamp(cur_f[:,segments_nro], min=0.001)
            '''
            
            #else:
                
            cur_f_inner = torch.exp(torch.index_select(cur_f,1,segments_nro))
            cur_f_inner_temp = torch.zeros(cur_f_inner.size(dim=0),cur_f_inner.size(dim=1),device = device)
            cur_f_inner_temp[0] = cur_f_inner[0]
            cur_f_inner_temp[1] = cur_f_inner[1]/mus_weight
            cur_f_inner_temp[2] = cur_f_inner[2]
            cur_f_inner_temp[3] = cur_f_inner[3]/mus_weight
            '''
            
            true_inner = torch.index_select(true,1,segments_nro)
            


            # Loss only with respect to the inner values
            
            return  cur_f, self.loss(cur_f_inner_temp, true_inner)

  

class LGS_EtoE(nn.Module):
       
       def __init__(self, loss,xsize,ysize,niter,combined=False,cropped=False,data_assisted=False):
           super().__init__()
           
           
           # Scalar values 
           self.xsize         =     xsize             # Grid size
           self.ysize         =     ysize
           self.combined      =     combined
           
           self.niter         =     niter             # LGS iterations             
           self.loss          =     loss              # Loss functions
           self.cropped       =     cropped
           self.data_assisted =     data_assisted

           
           if combined == True:
               mm = 6
               c_out = 3
           else:
               mm = 2
               c_out = 1 
               if self.data_assisted == True:
                   mm += 1
               
           for i in range(niter):
               iteration_mua = Iteration(mm,c_out)
               setattr(self, 'iteration_mua_{}'.format(i), iteration_mua)
               
               if combined == "partially" and i > 0:
                   iteration_mus = Iteration(mm+1,c_out)
               else:
                   iteration_mus = Iteration(mm,c_out)
               setattr(self, 'iteration_mus_{}'.format(i), iteration_mus)

           
           
       def forward(self,geom_info,cur_f,sample_info,mus_weight,device='cpu',cropped_indices=0,cur_data=0):      

           segments = sample_info.segments
           C = sample_info.C
           true = sample_info.true
           grad_init = sample_info.grad_init
           
           
           grad_mua_2 = torch.zeros(1,self.ysize*self.xsize,device = device)
           grad_mua = torch.zeros(1,self.ysize*self.xsize,device = device)
           grad_mus = torch.zeros(1,self.ysize*self.xsize,device = device)
 
           cur_mua_vec = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           cur_mus_vec = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           cur_mua_vec_2 = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           crop_indices = cropped_indices[:,0]
           
           for i in range(self.niter):
                                           
               if self.cropped == True:                             
                    cur_mua_vec[0,0,crop_indices] =  cur_f[0,:]
                    cur_mua_vec_2[0,0,crop_indices] =  cur_f[2,:]
                    cur_mus_vec[0,0,crop_indices] =  cur_f[1,:]
                    
               else:
                   
                    cur_mua_vec[0,0] = cur_f[0,:]
                    cur_mus_vec[0,0] = cur_f[1,:]
                    cur_mua_vec_2[0,0] = cur_f[2,:]
                   
      
                  
               if i == 0 :
                   
                   if self.cropped == True:
                       grad_mua[0,crop_indices] = grad_init[0]
                       grad_mua_2[0,crop_indices] = grad_init[2]
                       grad_mus[0,crop_indices] = grad_init[1]
                       
                   else:
                       grad_mua[0] = grad_init[0,:]
                       grad_mus[0] = grad_init[1,:]
                       grad_mua_2[0] = grad_init[2,:]
                       
               else:
                   
                   grad_mua_temp,grad_mus_temp,grad_mua_2_temp = FE.Get_grad_torch_EtoE(geom_info, cur_f,sample_info, device)

                   if self.cropped == True:
                        grad_mua[0,crop_indices] = grad_mua_temp
                        grad_mua_2[0,crop_indices] = grad_mua_2_temp
                        grad_mus[0,crop_indices] = grad_mus_temp
                        
                   else:
                        grad_mua = grad_mua_temp
                        grad_mus = grad_mus_temp
                        grad_mua_2 = grad_mua_2_temp



               cur_mua_grid = torch.transpose(torch.reshape(cur_mua_vec,(1,1,self.ysize,self.xsize)),2,3)              
               cur_mua_grid_2 = torch.transpose(torch.reshape(cur_mua_vec_2[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)              
               cur_mus_grid = torch.transpose(torch.reshape(cur_mus_vec[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)


               add_mua = torch.zeros(1,1,geom_info.n,device = device)
               add_mua_2 = torch.zeros(1,1,geom_info.n,device = device)
               add_mus = torch.zeros(1,1,geom_info.n,device = device) 
    
    
               grad_mus_grid = torch.transpose(torch.reshape(grad_mus,(self.ysize,self.xsize)),0,1)
               grad_mua_grid = torch.transpose(torch.reshape(grad_mua,(self.ysize,self.xsize)),0,1) 
               grad_mua_2_grid = torch.transpose(torch.reshape(grad_mua_2,(self.ysize,self.xsize)),0,1) 

               iteration_mua = getattr(self, 'iteration_mua_{}'.format(i))
               iteration_mus = getattr(self, 'iteration_mus_{}'.format(i))
               segments_nro = torch.squeeze(torch.nonzero(segments[0,:]))

               if self.combined == True:
                   cur_all = torch.cat([cur_mua_grid,cur_mua_grid_2,cur_mus_grid], dim=1)
                   grad_all=torch.cat([grad_mua_grid[None,None,:,:],grad_mua_2_grid[None,None,:,:],grad_mus_grid[None,None,:,:]], dim=1)
                   inp_all = iteration_mua(cur_all,grad_all)
                   inp_all = torch.flatten(torch.transpose(inp_all,2,3),2,3)
                   
                   if self.cropped == True:
                       inp_all = inp_all[:,:,:,crop_indices]
                       
                   inp_all = torch.index_select(inp_all,2,segments_nro)
                   add_mua[0,0,segments_nro] = inp_all[:,[0]]
                   add_mua_2[0,0,segments_nro] = inp_all[:,[1]]
                   add_mus[0,0,segments_nro] = inp_all[:,[2]]            
               
               else:
               
                   if self.data_assisted == True:
                       inp_mua = iteration_mua(cur_mua_grid,torch.cat([grad_mua_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       inp_mua_2 = iteration_mua(cur_mua_grid_2,torch.cat([grad_mua_2_grid[None,None,:,:],cur_data[:,[1]]],dim=1))
                   else:
                       inp_mua = iteration_mua(cur_mua_grid,grad_mua_grid[None,None,:,:])
                       inp_mua_2 = iteration_mua(cur_mua_grid_2,grad_mua_2_grid[None,None,:,:])
                   if self.combined =="partially" and i > 0:
                       if self.data_assisted == True:
                           inp_mus = iteration_mus(torch.cat([cur_mus_grid,cur_mua_grid], dim=1),torch.cat([grad_mus_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       else:
                           inp_mus = iteration_mus(torch.cat([cur_mus_grid,cur_mua_grid], dim=1),grad_mus_grid[None,None,:,:])

                   else:
                       if self.data_assisted == True:
                           inp_mus = iteration_mus(cur_mus_grid,torch.cat([grad_mus_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       else:
                           inp_mus = iteration_mus(cur_mus_grid,grad_mus_grid[None,None,:,:])

                   # To vectors
    
                   inp_mus = torch.flatten(torch.transpose(inp_mus,2,3),2,3)
                   inp_mua = torch.flatten(torch.transpose(inp_mua,2,3),2,3)
                   inp_mua_2 = torch.flatten(torch.transpose(inp_mua_2,2,3),2,3)
                   if self.cropped == True:
                       inp_mus = inp_mus[:,:,crop_indices]
                       inp_mua = inp_mua[:,:,crop_indices]
                       inp_mua_2 = inp_mua_2[:,:,crop_indices]
                   # Circle gradients 
    
                   inp_mua = torch.index_select(inp_mua,2,segments_nro)
                   inp_mua_2 = torch.index_select(inp_mua_2,2,segments_nro)
                   inp_mus = torch.index_select(inp_mus,2,segments_nro)
    
                   add_mua[0,0,segments_nro] = inp_mua
                   add_mua_2[0,0,segments_nro] = inp_mua_2
                   add_mus[0,0,segments_nro] = inp_mus


               cur_mua = cur_f[0] + 0.01*add_mua[0]
               cur_mus = cur_f[1] + 0.01*add_mus[0]
               cur_mua_2 = cur_f[2]+ 0.01*add_mua_2[0]
               

               cur_mus_2 = cur_mus*C
    
                   
               cur_f = torch.cat((cur_mua,cur_mus,cur_mua_2,cur_mus_2),dim=0)
               

               a = 0.1 
               b = 0.001 
               a_max = 3     
               b_max = 1.2 
                   
               if i < self.niter-1:
                   cur_f_inner = torch.index_select(cur_f,1,segments_nro)               
                   cur_f_inner_temp = torch.zeros(cur_f_inner.size(dim=0),cur_f_inner.size(dim=1),device = device)
                   cur_f_inner_temp[0] = torch.clamp(cur_f_inner[0], min=b,max=b_max)
                   cur_f_inner_temp[1] = torch.clamp(cur_f_inner[1], min=a,max=a_max)
                   cur_f_inner_temp[2] = torch.clamp(cur_f_inner[2], min=b,max=b_max)
                   cur_f_inner_temp[3] = torch.clamp(cur_f_inner[3], min=a,max=a_max)
                   cur_f[:,segments_nro] = cur_f_inner_temp
               else:
             
                   cur_f_inner = torch.index_select(cur_f,1,segments_nro)               


           cur_f_inner[1] = cur_f_inner[1]/mus_weight
           cur_f_inner[3] = cur_f_inner[3]/mus_weight
          
           true_inner = torch.index_select(true,1,segments_nro)


           return  cur_f, self.loss(cur_f_inner, true_inner)


class LGS_EtoE_model_error(nn.Module):
       
       def __init__(self, loss,xsize,ysize,niter,combined=False,cropped=False,data_assisted=False):
           super().__init__()
           
           
           # Scalar values 
           self.xsize         =     xsize             # Grid size
           self.ysize         =     ysize
           self.combined      =     combined
           
           self.niter         =     niter             # LGS iterations             
           self.loss          =     loss              # Loss functions
           self.cropped       =     cropped
           self.data_assisted =     data_assisted

           
           if combined == True:
               mm = 6
               c_out = 3
           else:
               mm = 2
               c_out = 1 
               if self.data_assisted == True:
                   mm += 1
               
           for i in range(niter):
               iteration_mua = Iteration(mm,c_out)
               setattr(self, 'iteration_mua_{}'.format(i), iteration_mua)
               
               if combined == "partially" and i > 0:
                   iteration_mus = Iteration(mm+1,c_out)
               else:
                   iteration_mus = Iteration(mm,c_out)
               setattr(self, 'iteration_mus_{}'.format(i), iteration_mus)

           
           
       def forward(self,geom_info,cur_f,sample_info,mus_weight,device='cpu',cropped_indices=0,cur_data=0):      

           segments = sample_info.segments
           C = sample_info.C
           true = sample_info.true
           grad_init = sample_info.grad_init
           
           
           grad_mua_2 = torch.zeros(1,self.ysize*self.xsize,device = device)
           grad_mua = torch.zeros(1,self.ysize*self.xsize,device = device)
           grad_mus = torch.zeros(1,self.ysize*self.xsize,device = device)
 
           cur_mua_vec = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           cur_mus_vec = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           cur_mua_vec_2 = torch.zeros(1,1,self.ysize*self.xsize,device = device)
           crop_indices = cropped_indices[:,0]
           
           for i in range(self.niter):
                                           
               if self.cropped == True:                             
                    cur_mua_vec[0,0,crop_indices] =  cur_f[0,:]
                    cur_mua_vec_2[0,0,crop_indices] =  cur_f[2,:]
                    cur_mus_vec[0,0,crop_indices] =  cur_f[1,:]
                    
               else:
                   
                    cur_mua_vec[0,0] = cur_f[0,:]
                    cur_mus_vec[0,0] = cur_f[1,:]
                    cur_mua_vec_2[0,0] = cur_f[2,:]
                   
      
                  
               if i == 0 :
                   
                   if self.cropped == True:
                       grad_mua[0,crop_indices] = grad_init[0]
                       grad_mua_2[0,crop_indices] = grad_init[2]
                       grad_mus[0,crop_indices] = grad_init[1]
                       
                   else:
                       grad_mua[0] = grad_init[0,:]
                       grad_mus[0] = grad_init[1,:]
                       grad_mua_2[0] = grad_init[2,:]
                       
               else:
                   
                   grad_mua_temp,grad_mus_temp,grad_mua_2_temp = FE.Get_grad_torch_EtoE(geom_info, cur_f,sample_info, device)

                   if self.cropped == True:
                        grad_mua[0,crop_indices] = grad_mua_temp
                        grad_mua_2[0,crop_indices] = grad_mua_2_temp
                        grad_mus[0,crop_indices] = grad_mus_temp
                        
                   else:
                        grad_mua = grad_mua_temp
                        grad_mus = grad_mus_temp
                        grad_mua_2 = grad_mua_2_temp



               cur_mua_grid = torch.transpose(torch.reshape(cur_mua_vec,(1,1,self.ysize,self.xsize)),2,3)              
               cur_mua_grid_2 = torch.transpose(torch.reshape(cur_mua_vec_2[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)              
               cur_mus_grid = torch.transpose(torch.reshape(cur_mus_vec[None,None,:,:],(1,1,self.ysize,self.xsize)),2,3)


               add_mua = torch.zeros(1,1,geom_info.n,device = device)
               add_mua_2 = torch.zeros(1,1,geom_info.n,device = device)
               add_mus = torch.zeros(1,1,geom_info.n,device = device) 
    
    
               grad_mus_grid = torch.transpose(torch.reshape(grad_mus,(self.ysize,self.xsize)),0,1)
               grad_mua_grid = torch.transpose(torch.reshape(grad_mua,(self.ysize,self.xsize)),0,1) 
               grad_mua_2_grid = torch.transpose(torch.reshape(grad_mua_2,(self.ysize,self.xsize)),0,1) 

               iteration_mua = getattr(self, 'iteration_mua_{}'.format(i))
               iteration_mus = getattr(self, 'iteration_mus_{}'.format(i))
               segments_nro = torch.squeeze(torch.nonzero(segments[0,:]))

               if self.combined == True:
                   cur_all = torch.cat([cur_mua_grid,cur_mua_grid_2,cur_mus_grid], dim=1)
                   grad_all=torch.cat([grad_mua_grid[None,None,:,:],grad_mua_2_grid[None,None,:,:],grad_mus_grid[None,None,:,:]], dim=1)
                   inp_all = iteration_mua(cur_all,grad_all)
                   inp_all = torch.flatten(torch.transpose(inp_all,2,3),2,3)
                   
                   if self.cropped == True:
                       inp_all = inp_all[:,:,:,crop_indices]
                       
                   inp_all = torch.index_select(inp_all,2,segments_nro)
                   add_mua[0,0,segments_nro] = inp_all[:,[0]]
                   add_mua_2[0,0,segments_nro] = inp_all[:,[1]]
                   add_mus[0,0,segments_nro] = inp_all[:,[2]]            
               
               else:
               
                   if self.data_assisted == True:
                       inp_mua = iteration_mua(cur_mua_grid,torch.cat([grad_mua_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       inp_mua_2 = iteration_mua(cur_mua_grid_2,torch.cat([grad_mua_2_grid[None,None,:,:],cur_data[:,[1]]],dim=1))
                   else:
                       inp_mua = iteration_mua(cur_mua_grid,grad_mua_grid[None,None,:,:])
                       inp_mua_2 = iteration_mua(cur_mua_grid_2,grad_mua_2_grid[None,None,:,:])
                   if self.combined =="partially" and i > 0:
                       if self.data_assisted == True:
                           inp_mus = iteration_mus(torch.cat([cur_mus_grid,cur_mua_grid], dim=1),torch.cat([grad_mus_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       else:
                           inp_mus = iteration_mus(torch.cat([cur_mus_grid,cur_mua_grid], dim=1),grad_mus_grid[None,None,:,:])

                   else:
                       if self.data_assisted == True:
                           inp_mus = iteration_mus(cur_mus_grid,torch.cat([grad_mus_grid[None,None,:,:],cur_data[:,[0]]],dim=1))
                       else:
                           inp_mus = iteration_mus(cur_mus_grid,grad_mus_grid[None,None,:,:])

                   # To vectors
    
                   inp_mus = torch.flatten(torch.transpose(inp_mus,2,3),2,3)
                   inp_mua = torch.flatten(torch.transpose(inp_mua,2,3),2,3)
                   inp_mua_2 = torch.flatten(torch.transpose(inp_mua_2,2,3),2,3)
                   if self.cropped == True:
                       inp_mus = inp_mus[:,:,crop_indices]
                       inp_mua = inp_mua[:,:,crop_indices]
                       inp_mua_2 = inp_mua_2[:,:,crop_indices]
                   # Circle gradients 
    
                   inp_mua = torch.index_select(inp_mua,2,segments_nro)
                   inp_mua_2 = torch.index_select(inp_mua_2,2,segments_nro)
                   inp_mus = torch.index_select(inp_mus,2,segments_nro)
    
                   add_mua[0,0,segments_nro] = inp_mua
                   add_mua_2[0,0,segments_nro] = inp_mua_2
                   add_mus[0,0,segments_nro] = inp_mus


               cur_mua = cur_f[0] + 0.01*add_mua[0]
               cur_mus = cur_f[1] + 0.01*add_mus[0]
               cur_mua_2 = cur_f[2]+ 0.01*add_mua_2[0]
               

               cur_mus_2 = cur_mus*C
    
                   
               cur_f = torch.cat((cur_mua,cur_mus,cur_mua_2,cur_mus_2),dim=0)
               

               a = 0.1 
               b = 0.001 
               a_max = 3     
               b_max = 1.2 
                   
               if i < self.niter-1:
                   cur_f_inner = torch.index_select(cur_f,1,segments_nro)               
                   cur_f_inner_temp = torch.zeros(cur_f_inner.size(dim=0),cur_f_inner.size(dim=1),device = device)
                   cur_f_inner_temp[0] = torch.clamp(cur_f_inner[0], min=b,max=b_max)
                   cur_f_inner_temp[1] = torch.clamp(cur_f_inner[1], min=a,max=a_max)
                   cur_f_inner_temp[2] = torch.clamp(cur_f_inner[2], min=b,max=b_max)
                   cur_f_inner_temp[3] = torch.clamp(cur_f_inner[3], min=a,max=a_max)
                   cur_f[:,segments_nro] = cur_f_inner_temp
               else:
             
                   cur_f_inner = torch.index_select(cur_f,1,segments_nro)               


           cur_f_inner[1] = cur_f_inner[1]/mus_weight
           cur_f_inner[3] = cur_f_inner[3]/mus_weight
          
           true_inner = torch.index_select(true,1,segments_nro)


           return  cur_f, self.loss(cur_f_inner, true_inner)




        
       
def resBlock(in_channels, conv_channels,conv_out):
       return nn.Sequential(
           
          # 3 conv layers
          
          # bias = False if vonv2s followed by batchnorm2d
          # See: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
          nn.Conv2d(in_channels, conv_channels, 3, padding=1,bias = True),
          nn.GroupNorm(8, conv_channels),
          #nn.BatchNorm2d(conv_channels),       
          nn.ReLU(inplace=True),
          nn.Conv2d(conv_channels, conv_channels, 3, padding=1,bias = True),
          nn.GroupNorm(8, conv_channels),
          #nn.BatchNorm2d(conv_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(conv_channels, conv_channels, 3, padding=1, bias = True),
          #nn.BatchNorm2d(conv_channels),
          nn.GroupNorm(8, conv_channels),
          nn.ReLU(inplace=True),
          
          # Compressing layer
          # To original channels
          nn.Conv2d(conv_channels, conv_out, 3, padding=1),
          )

class Iteration(nn.Module):
       def __init__(self,c_in,c_out):
           super().__init__()
           self.resBlock = resBlock(c_in, 32,c_out)

       def forward(self,cur,cur_grad):
              
           inp = torch.cat([cur, cur_grad], dim=1)

           inp = self.resBlock(inp)

           # Iteration update
           return inp
       
        
       
        
class UNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv(n_in, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

       
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose2d(64,32,2,stride=2,padding=0)
        

        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        self.conv_last = nn.Conv2d(32, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x,loss,true,segments,seg_inv):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        
        
        x = self.xUp2(conv3)                
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        update = torch.flatten(torch.transpose(update,2,3),2,3)
        
        update_inner = update[0,0,segments[:,0]]
        true_inner = true[0,segments[:,0]]
                
        update[:,:,seg_inv] = 0
        
        
        return update, loss(update_inner, true_inner)
 
    
 
def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True))
    

