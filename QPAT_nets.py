import torch
from torch import nn
import QPAT_util as Qutil


class Iteration(nn.Module):
    def __init__(self,n):
        super().__init__()
        
        self.resBlock = resBlock(2, 32)
        #self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.n = n

    def forward(self,cur,cur_grad):
           
        inp = torch.cat([cur, cur_grad], dim=1)

        inp = self.resBlock(inp)

        # Iteration update (small scaling factor to avoid large values with untrained network)
        return cur + 0.1*inp
    
    
    
def resBlock(in_channels, conv_channels):
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
       nn.Conv2d(conv_channels, 1, 3, padding=1),
       
       )


class LGS_EtoE(nn.Module):
    def __init__(self, niter,loss,n,xsize,ysize):
        super().__init__()
        
        # Scalar values 
        self.xsize         =     xsize             # Grid size
        self.ysize         =     ysize
        self.n             =     n                 # number of unknowns
        self.loss          =     loss              # Loss functions
        self.niter         =     niter


        for i in range(niter):
            iteration_mua = Iteration(self.n)
            setattr(self, 'iteration_mua_{}'.format(i), iteration_mua)
            iteration_mus = Iteration(self.n)
            setattr(self, 'iteration_mus_{}'.format(i), iteration_mus)
        
             
    def forward(self,current_f,data,true,geom,bSize=1,ns=2,Le_vec=0,init_grad=[],device='cuda'):      

        
        current_f.requires_grad_(True)
        current_mua = current_f[:,[0],:,:]
        current_mus = current_f[:,[1],:,:]
        if geom.solver == 2:
            Hes_prev = torch.eye(2*self.n,device = torch.device(device))  
            
        for i in range(self.niter):
            
            current_mus = torch.flatten(torch.transpose(current_mus,2,3),2,3)
            current_mua = torch.flatten(torch.transpose(current_mua,2,3),2,3)
            if geom.solver == 2:
                x_prev = torch.cat((current_mua[0],current_mus[0]),dim=1)
                
            if i == 0:
                dx_mua = init_grad[:,0]
                dx_mus = init_grad[:,1]
                
                if geom.solver == 2:
                    Hes_prev = torch.eye(2*self.n,device = torch.device(device))
                    grad_prev = init_grad               
              
            else:
                if geom.solver == 2:            
                    dx_mua,dx_mus,Hes_prev,grad_prev = Qutil.Get_grad_torch_EtoE(geom, current_mua, current_mus, data, bSize, ns, Le_vec,device,i,Hes_prev,grad_prev,x_prev)                    
                else:
                    dx_mua,dx_mus = Qutil.Get_grad_torch_EtoE(geom, current_mua, current_mus, data, bSize, ns, Le_vec,device,i)               
                    #dx_mua,dx_mus,Hes_prev,grad_prev = checkpoint(Get_grad_torch,self, current_mua, current_mus, data, bSize, data_means, ns, Le_vec,qvec,device,i,Hes_prev,grad_prev,x_prev)

            # To grid form

            if geom.solver == 2:
                x_prev = torch.cat((current_mua[0],current_mus[0]),dim=1)

            current_mua = torch.transpose(torch.reshape(current_mua,(1,1,self.ysize,self.xsize)),2,3)
            current_mus = torch.transpose(torch.reshape(current_mus,(1,1,self.ysize,self.xsize)),2,3)
             
            dx_mus = torch.transpose(torch.reshape(dx_mus,(1,1,self.ysize,self.xsize)),2,3)
            dx_mua = torch.transpose(torch.reshape(dx_mua,(1,1,self.ysize,self.xsize)),2,3) 
            
            iteration_mua = getattr(self, 'iteration_mua_{}'.format(i))
            iteration_mus = getattr(self, 'iteration_mus_{}'.format(i))
            
            current_mua = iteration_mua(current_mua,dx_mua)
            current_mus = iteration_mus(current_mus,dx_mus)
        

        current_f = torch.exp(torch.cat((current_mua,current_mus),dim=1))

        return current_f, self.loss(current_f, true)
            
    
    
    
    
class LGS(nn.Module):
    
    def __init__(self, loss,xsize,ysize,n):
        super().__init__()
        
        # Scalar values 
        self.xsize         =     xsize             # Grid size
        self.ysize         =     ysize
        self.loss          =     loss


        for i in range(1):
            iteration_mua = Iteration(n)
            setattr(self, 'iteration_mua_{}'.format(i), iteration_mua)
            iteration_mus = Iteration(n)
            setattr(self, 'iteration_mus_{}'.format(i), iteration_mus)
        
        
    def forward(self,current_f,true,dx):      

        
        #current_f.requires_grad_(True)
        
        with torch.no_grad():
            current_mua = current_f[:,[0],:,:]
            current_mus = current_f[:,[1],:,:]
                     
            dx_mua = dx[:,[0]]
            dx_mus = dx[:,[1]]
    
            
        iteration_mua = getattr(self, 'iteration_mua_{}'.format(0))
        iteration_mus = getattr(self, 'iteration_mus_{}'.format(0))
        
        current_mua = iteration_mua(current_mua,dx_mua)
        current_mus = iteration_mus(current_mus,dx_mus)

     
        current_f = torch.cat((current_mua,current_mus),dim=1)

        return current_f, self.loss(torch.exp(current_f), true)
    
    
    
    
    
class LGS_separate(nn.Module):
    
    
    def __init__(self, loss,xsize,ysize,n):
        super().__init__()
        
        # Scalar values 
        self.xsize         =     xsize             # Grid size
        self.ysize         =     ysize
        self.loss          =     loss

        self.iteration = Iteration(n)
            

               
    def forward(self,current_f,true,dx):      
        
        current_f = self.iteration(current_f,dx)

        return current_f, self.loss(torch.exp(current_f), true)
    
    
            