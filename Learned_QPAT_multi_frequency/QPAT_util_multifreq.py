
import QPAT_FEM_multifreq as FE


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io



class DataSet(object):

  def __init__(self, data, true,initial,segments,C,prior_vals):
    """Construct a DataSet"""
    
    self._num_examples = true.size(dim=1)
    self._freq_nro = true.size(dim=2)
    self._data = data
    self._true = true
    self._initial = initial
    #self._temp_grad = np.zeros(true.shape)
    self._segments = segments
    self._C = C
    self._prior_vals = prior_vals
  @property
  def segments(self):
    return self._segments



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
  def C(self):
    return self._C

  @property
  def prior_vals(self):
    return self._prior_vals

  def next_batch(self, batch_size,sample_num,freq_nums=0):
    """Return the next `batch_size` examples from this data set."""
    rng = np.random.default_rng()
    freq_num = int(2*rng.choice(int(self._freq_nro/2), size=1, replace=False))

  
    return self._data[sample_num,freq_num:freq_num+2,:], self._true[:,sample_num,freq_num:freq_num+2,:], self._initial[:,sample_num,freq_num:freq_num+2,:],self._segments[sample_num,:],freq_num



def Get_experimental_data(path,path_test,simulated,freqs,augment,outliers=False,form_spatials=False):
    
    dir_list = os.listdir(path)
    dir_list_test = os.listdir(path_test)
    if outliers != False:
        dir_list_outliers = os.listdir(outliers)
        sample_names_outliers = []


    sample_names = []
    sample_names_test = []
    for i in range(len(dir_list)):
        sample_names.append(dir_list[i].partition("_")[0])
        if i < len(dir_list_outliers):
            sample_names_outliers.append(dir_list_outliers[i].partition("_")[0])
        if i < len(dir_list_test):
            sample_names_test.append(dir_list_test[i].partition("_")[0])
           
    dir_list_test = np.sort(dir_list_test)
    dir_list_outliers = np.sort(dir_list_outliers)
    dir_list = np.sort(dir_list)

    sample_names = np.unique(sample_names)
    sample_names_test = np.unique(sample_names_test)
    
    sample_names_outliers = np.unique(sample_names_outliers)
    # 24 training samples with freqs frequencies
    train_set = {}
    test_set = {}
    
    
    nodes = 288*288 # experimental data resolution
    if augment == True:
        sample_nro = int(2*len(sample_names))
        orig_sample_train = len(sample_names)

    else:
        sample_nro = len(sample_names)
    if outliers == False:
        test_sample_nro = len(sample_names_test)
    else:
        test_sample_nro = len(sample_names_test) + len(sample_names_outliers) 

    data = torch.zeros(sample_nro,freqs,nodes)
    abs_coeffs = torch.zeros(sample_nro,freqs,nodes)
    scat_coeffs = torch.zeros(sample_nro,freqs,nodes)
    segmentation = torch.zeros(sample_nro,freqs,nodes)


    data_test = torch.zeros(test_sample_nro,freqs,nodes)
    abs_coeffs_test = torch.zeros(test_sample_nro,freqs,nodes)
    scat_coeffs_test = torch.zeros(test_sample_nro,freqs,nodes)
    segmentation_test = torch.zeros(test_sample_nro,freqs,nodes)
    
    
    ind = 0
    freq = 0
    
    for i in range(len(dir_list)):
    
        sample_name = path + dir_list[i]
    
        stats = np.load(sample_name)

        if simulated == True:
            data[ind,freq,:] =  torch.from_numpy(np.transpose(stats["fluence"]).flatten())
        else:
            data[ind,freq,:] =  torch.from_numpy(np.transpose(stats["features_das"]).flatten())

        scat_coeffs[ind,freq,:] =  torch.from_numpy(np.transpose(stats["musp"]).flatten())
        abs_coeffs[ind,freq,:] =  torch.from_numpy(np.transpose(stats["mua"]).flatten()) 
        segmentation[ind,freq,:] =  torch.from_numpy(np.transpose(stats["segmentation"]).astype(bool).flatten()) 
        
        if augment == True:
            if simulated == True:
                data[ind+orig_sample_train,freq,:] =  torch.from_numpy(np.flip(np.transpose(stats["fluence"]),0).flatten())
            else:
                data[ind+orig_sample_train,freq,:] =  torch.from_numpy(np.flip(np.transpose(stats["features_das"]),0).flatten())

            scat_coeffs[ind+orig_sample_train,freq,:] =  torch.from_numpy(np.flip(np.transpose(stats["musp"]),0).flatten())
            abs_coeffs[ind+orig_sample_train,freq,:] =  torch.from_numpy(np.flip(np.transpose(stats["mua"]),0).flatten()) 
            segmentation[ind+orig_sample_train,freq,:] =  torch.from_numpy(np.flip(np.transpose(stats["segmentation"]),0).astype(bool).flatten()) 
            
                  
        ###########################
        ## DOWNLOAD TEST SAMPLES ##
        ###########################
        
        if ind < test_sample_nro:
            if ind < len(sample_names_test):
                sample_name_test = path_test + dir_list_test[i]
            else:
                sample_name_test = outliers + dir_list_outliers[i-len(dir_list_test)]

            stats_test = np.load(sample_name_test)
            if simulated == True:
                data_test[ind,freq,:] = torch.from_numpy(np.transpose(stats_test["fluence"]).flatten())
            else:
                data_test[ind,freq,:] =  torch.from_numpy(np.transpose(stats_test["features_das"]).flatten())
                
            scat_coeffs_test[ind,freq,:] =  torch.from_numpy(np.transpose(stats_test["musp"]).flatten())
            abs_coeffs_test[ind,freq,:] =  torch.from_numpy(np.transpose(stats_test["mua"]).flatten()) 
            segmentation_test[ind,freq,:] =  torch.from_numpy(np.transpose(stats_test["segmentation"]).astype(bool).flatten()) 
            
    
    
            
        freq += 1
        
        
        if (i + 1) % freqs == 0: 
            ind += 1
            freq = 0
            
    abs_coeffs/=10   
    abs_coeffs_test/=10
    scat_coeffs/=10    
    scat_coeffs_test/=10 
    
    train_set["fluence"] = data    
    test_set["fluence"] = data_test 
    train_set["mua"] = abs_coeffs    
    test_set["mua"] =  abs_coeffs_test
    
    train_set["mus"] = scat_coeffs   
    test_set["mus"] = scat_coeffs_test
    train_set["segmentation"] = segmentation    
    test_set["segmentation"] = segmentation_test
    
    


    #p.savez("data/train_set.npz",segmentation=segmentation,mua=abs_coeffs,mus=scat_coeffs,fluence=data)
    #np.savez("data/test_set.npz",segmentation=segmentation_test,mua=abs_coeffs_test,mus=scat_coeffs_test,fluence=data_test)





    return train_set,test_set

            
           
def Interpolate_experimental(dataset,geom,setname,segmentation_crop,freqs,spatial_prior=False):            
            
    if os.path.isfile(setname + ".pt"):
        intp_dataset = torch.load(setname+'.pt') 

    else:   
        coords = torch.from_numpy(geom["orig_coords"]).float()
        coords_intp = torch.from_numpy(geom["coords"]).float()
        n = geom["n"]
        
        #dh = torch.abs(coords_intp[1,0]-coords_intp[1,1])
        dh_init = 30.72/(288-1)
        
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
        samples = len(dataset["mua"][:,0,0])
        data_intp = torch.zeros((samples,freqs,n))
        mua_intp = torch.zeros((samples,freqs,n))
        mus_intp = torch.zeros((samples,freqs,n))
        init_mua_intp = torch.zeros((samples,freqs,n))
        init_mus_intp = torch.zeros((samples,freqs,n))


        intp_segmentation = torch.zeros((samples,freqs,n),dtype=int)
        freq_nro = freqs
    
        for i in range(samples):
            print("Sample "+str(i+1)+" being interpolated")
            for j in range(freq_nro):
                                
                orig_init_mua = dataset["mua"][i,j].clone()
                orig_init_mus = dataset["mus"][i,j].clone()
                bkg_orig_abs = torch.median(dataset["mua"][i,j,dataset["segmentation"][i,j].to(torch.bool)])
                bkg_orig_scat = torch.median(dataset["mus"][i,j,dataset["segmentation"][i,j].to(torch.bool)])
                if spatial_prior == False:
                    orig_init_mua[dataset["segmentation"][i,j].to(torch.bool)] = bkg_orig_abs
                else:
                    orig_init_mua[dataset["segmentation"][i,j].to(torch.bool)] = bkg_orig_abs*dataset['spatial_priors'][i,j,dataset["segmentation"][i,j].to(torch.bool)]
                
                
                orig_init_mus[dataset["segmentation"][i,j].to(torch.bool)] = bkg_orig_scat

                for t in range(n):
                    data_cur = torch.index_select(dataset["fluence"][i,j],0,indices[t])
                    mus_cur = torch.index_select(dataset["mus"][i,j],0,indices[t])
                    mua_cur = torch.index_select(dataset["mua"][i,j],0,indices[t])
                    
                    init_mus_cur = torch.index_select(orig_init_mus,0,indices[t])
                    init_mua_cur = torch.index_select(orig_init_mua,0,indices[t])
                    
                    w_sum = torch.sum(weights[t])
                    # The coeffs are also scaled from 1/cm to 1/mm
                    mus_intp[i,j,t] = torch.sum(torch.multiply(weights[t],mus_cur))/w_sum
                    mua_intp[i,j,t] = torch.sum(torch.multiply(weights[t],mua_cur))/w_sum  
                    data_intp[i,j,t] = torch.sum(torch.multiply(weights[t],data_cur))/w_sum

                    init_mus_intp[i,j,t] = torch.sum(torch.multiply(weights[t],init_mus_cur))/w_sum
                    init_mua_intp[i,j,t] = torch.sum(torch.multiply(weights[t],init_mua_cur))/w_sum    
           


                    if (dataset["segmentation"][i,j][min_ind[t]] != 0 and mua_intp[i,j,t]>0.25*bkg_orig_abs and mus_intp[i,j,t]>0.25*bkg_orig_scat) or (dataset["segmentation"][i,j][min_ind[t]] == 0 and mua_intp[i,j,t]>bkg_orig_abs*0.5 and mus_intp[i,j,t]>0.5*bkg_orig_scat) :
                        intp_segmentation[i,j,t] = 1
                #print(str(torch.sum(intp_segmentation)))
                a = (intp_segmentation[i,j,:]).numpy().astype(bool)
                segmentation_crop += a
                segmentation_crop = segmentation_crop.astype(bool)                
                

        scipy.io.savemat("crop_files/crop_location_nodes_"+str(n)+".mat", {'segmentation_crop_interpolated': segmentation_crop})
  
                        
        intp_dataset = torch.zeros((6,samples,freqs,n))
        
        data_intp = torch.mul(data_intp,mua_intp)       
        intp_dataset[0] = data_intp;
        intp_dataset[1] = mua_intp
        intp_dataset[2] = mus_intp;
        intp_dataset[3] = intp_segmentation
        intp_dataset[4] = init_mua_intp
        intp_dataset[5] = init_mus_intp

        # save datasets
        torch.save(intp_dataset,setname+".pt") 
        #scipy.io.savemat(setname+".mat", {'intp_dataset': intp_dataset.cpu().numpy()})
    
    
    return intp_dataset,segmentation_crop
    
def Get_geom_info(geom_info):
    geom_info = geom_info['geom']
    geom = {}

    # Convert matlab struct to python form

    for i in range(len(geom_info.dtype.descr)):

        field = geom_info.dtype.descr[i][0]
        char_type = geom_info[field][0,0].dtype
        info = geom_info[field][0,0]
        if field == "b_source":
            print(field)

        if char_type == "uint16":
            geom[field] = info.astype(int)
        else:
            geom[field] = info  


    return geom


def Form_Weight_matrix(data,samples,segmentation,add_noise,freqs):
    
    # Set noise level
    noiseLev = 0.01 
    n = len(data[0,0,:])
    Le = torch.zeros((samples,freqs,n))

    for idx in range(samples):
        segments = torch.nonzero(segmentation[idx,:])
        for f in range(freqs):
            if add_noise == True:
                data[idx,f,:] += torch.random.normal(size=[n])*data[idx,f,:]*noiseLev
                         
            Le[idx,f,:] = 1/torch.mean(torch.abs((data[idx,f,segments]*noiseLev)))
            
    mean_Le = torch.mean(Le[:,:,0])

    return mean_Le
    
def Get_mean_set(coeffs,segments):

    circle_coeffs = torch.mul(coeffs,segments)
    sum_coeffs = torch.sum(torch.flatten(circle_coeffs))
    avg_coef = sum_coeffs / torch.sum(torch.flatten(segments))


    return avg_coef    


def Visualize_samples(values,fig_name,name_of_coeffs,mus_weight=1):
    

    samples = 1
    
    values = np.transpose(values,(0,1,3,2))

    fig,ax = plt.subplots(samples,2)
    
    set_name = name_of_coeffs
    for s in range(samples):         
        
        plt.subplot(samples,2,2*s+1)
        plt.imshow(values[s,0,:,:])
        plt.colorbar()
        plt.title(set_name+' (a)')

        plt.axis('off')
        plt.subplot(samples,2,2*s+2)
        plt.imshow(values[s,1,:,:]*mus_weight)
        plt.title(set_name+' (s)')
        plt.colorbar()
        plt.axis('off')

    fig.savefig(fig_name)



def Set_initial_values(coeffs,true,segments,scaled,interpolated_init,freqs):
    
    init_coeffs = torch.zeros(coeffs.shape)
    if interpolated_init == False:
        for i in range(len(coeffs[:,0,0])):
            for f in range(freqs):
                init_coeffs[i,f] = torch.mul(torch.abs(segments[i,f]-1),true[i,f]) + segments[i,f]
                if scaled == True:
                    init_coeffs[i,f] = torch.log(init_coeffs[i,f])
                    #coeffs[i,f] = torch.log(coeffs[i,f])
    else:
        for i in range(len(coeffs[:,0,0])):
            for f in range(freqs):
                init_coeffs[i,f] = coeffs[i,f]

            
    return init_coeffs
    
    
    

# Functions to compute statistics

def summary_image_impl(writer, name, tensor, it):
    image = tensor[0, 0]
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')



def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)      
    

def summaries(writer, result, true, loss, it,size_grid,segments,cropped_indices=0,mus_weight=15,positivity = 0):

    
    fixed_scat = False
    if result.size(dim=0) != 2:
        fixed_scat = True
        result_grid = torch.transpose(torch.reshape(result[:,0,-1],(1,size_grid,size_grid)),1,2)      
    else:
        result_grid = torch.transpose(torch.reshape(result[:,0,-1],(2,size_grid,size_grid)),1,2)  

    true_grid = torch.transpose(torch.reshape(true[:,0,-1],(2,size_grid,size_grid)),1,2)  


    if torch.is_tensor(cropped_indices):
        result = result[:,:,:,cropped_indices[:,0]]
        true = true[:,:,:,cropped_indices[:,0]]

     

    if positivity != "exp":
        if fixed_scat == False:
            result[1] = result[1]/mus_weight
            rel_scat= torch.norm(((result[1,:,:,segments]) - (true[1,:,:,segments]))  ) / torch.norm(((true[1,:,:,segments]))  )


        relative = torch.norm((result[:,:,:,segments] - true[:,:,:,segments])) / torch.norm((true[:,:,:,segments]))
        rel_abs = torch.norm(((result[0,:,:,segments]) - (true[0,:,:,segments]))  ) / torch.norm(((true[0,:,:,segments]))  )

    else:
        
        scaled_result = torch.exp(result.detach().clone())
        scaled_result[1] = scaled_result[1]/mus_weight
        scaled_true = true.detach().clone()


        
        
        relative = torch.norm(torch.exp(result) - true)/torch.norm(true[:,:,:,segments])
        rel_abs = torch.norm((scaled_result[0,:,:,segments] - (scaled_true[0,:,:,segments]))  ) / torch.norm(scaled_true[0,:,:,segments])
        if fixed_scat == False:
            rel_scat= torch.norm((scaled_result[1,:,:,segments] - (scaled_true[1,:,:,segments]))  ) / torch.norm(scaled_true[1,:,:,segments])
            result_grid = torch.transpose(torch.reshape(torch.exp(result[:,0,-1]),(2,size_grid,size_grid)),1,2)  
            
    writer.add_scalar('loss', loss, it)


    writer.add_scalar('relative', relative, it)

    writer.add_scalar('relative abs', rel_abs, it)

    
    summary_image(writer, 'Mua out', result_grid[None,[0],:,:], it)
    summary_image(writer, 'True Mua', true_grid[None,[0],:,:], it)
    
    if fixed_scat == False:   
        summary_image(writer, 'Mus out', result_grid[None,[1],:,:], it)
        summary_image(writer, 'True Mus', true_grid[None,[1],:,:], it)
        writer.add_scalar('relative scat', rel_scat, it)

    
    
    
    
class geom_specs: 
       def __init__(self,B,
                    n,xsize,ysize,elem,V_fff,V_fdd,solver,JJ,det_J_t,indices_elem,node_matrix,node_vector,indice_matrix,M_base,cropped_indices = 0):
           
           
           # Scalar values 

           self.n             =     n                 # number of unknowns
           self.solver        =     solver            # GN or GD

           self.node_vector   =     node_vector
       
           self.B             =     B                 # Constant part of sysmat

           self.V_fff         =     V_fff             # Sparse mass (integral) matrices
           self.V_fdd         =     V_fdd
           
           #self.coords        =     coords            # Node coordinates
           self.elem          =     elem              # Element node indices
           self.node_matrix   =     node_matrix
           
           
           # Auxiliary variables for FEM system matrix computation
           self.det_J_t       =     det_J_t
           self.JJ            =     JJ
           #self.indices_elem  =     indices_elem
           self.indice_matrix =     indice_matrix

           # Indices of water cropped nodes in the square
           self.cropped_indices   =     cropped_indices
           self.M_base        = M_base
           

class sample_info:
    def __init__(self,qvec,stds,data,prior_vals,C,true,segments,init_gradients):
           
       self.qvec = qvec
       self.stds = stds
       self.data = data
       self.prior_vals = prior_vals
       self.C = C
       self.true = true
       self.segments = segments
       self.grad_init = init_gradients
       
       
    def Get_sample_info(self,s_ind,f_ind,device):
           
        
        qvec_single = self.qvec[:,f_ind:f_ind+2].to(device)
        C_single = self.C[[s_ind],f_ind,f_ind+1].to(device)
        stds_single = self.stds[:,[s_ind],f_ind:f_ind+2].to(device)
        data_single = self.data[[s_ind],f_ind:f_ind+2].to(device)
        prior_vals_single = self.prior_vals[:,[s_ind],f_ind:f_ind+2].to(device)
        true_single = torch.cat((self.true[:,s_ind,f_ind],self.true[:,s_ind,f_ind+1]),dim=0).to(device)
        segments_single = self.segments[[s_ind]].to(device)
        grad_init =     self.grad_init[:,[s_ind],int(f_ind/2)].to(device)
        
        

        single_sample_info = sample_info(qvec_single,stds_single,data_single,prior_vals_single,C_single,true_single,segments_single,grad_init)
        
        return single_sample_info



def Form_new_prior(priors,mua,mus,segments,device):
    
    new_priors = torch.zeros(priors.size(),device=device)
    for s in range(priors.size(dim=1)):
        seg = torch.nonzero(segments[s,:])
        seg = seg[:,0]

        for f in range(priors.size(dim=2)):
            #if positivity == "exp":
            #    a = torch.mean(priors[0,s,f,seg])/torch.mean(torch.exp(mua[s,f,seg]))
            #    b = torch.mean(priors[1,s,f,seg])/torch.mean(torch.exp(mus[s,f,seg]))
            #else:
            a = torch.mean(priors[0,s,f,seg])/torch.mean(mua[s,f,seg])
            b = torch.mean(priors[1,s,f,seg])/torch.mean(mus[s,f,seg])
           
            new_priors[0,s,f,seg] = priors[0,s,f,seg]/a
            new_priors[1,s,f,seg] = priors[1,s,f,seg]/b

    
    return new_priors






def BAE_grouped(H,H_acc,segments,device):
    
    n = H.size(dim=2)
    samples = H.size(dim=0)
    freqs = H.size(dim=1)
    mean_mat = torch.zeros(n,device=device)
    mean_hit = torch.zeros(n,device=device)
    var_mat = torch.zeros(n,n,device=device)
    dif_mat = torch.zeros(samples,freqs,n,device=device)
    var_hit = torch.zeros(n,n,device=device)
    # Determine nodewise means
    seg_sum = torch.zeros(n,device=device)
    for i in range(samples):
        seg = torch.nonzero(segments[i,:])
        seg = seg[:,0]
        seg_sum[seg]+=1
    
    seg_union = torch.nonzero(seg_sum == samples)
    seg = seg_union[:,0]
    
    for i in range(samples):
        for f in range(freqs):
            mean_mat[seg] += H_acc[i,f,seg].to(device)-H[i,f,seg]
            dif_mat[i,f,seg] = H_acc[i,f,seg].to(device)-H[i,f,seg]
            
    mean_mat /= samples
    
    # Determine nodewise samplevariance
    for i in range(samples):               
        for f in range(freqs):
            var_mat += (dif_mat[i,f,None].T - mean_mat[:,None]) @ (dif_mat[None,i,f] - mean_mat[None,:])
            

    var_mat /= (samples-1)    
    


    return var_mat,mean_mat

def PriorOrnsteinUhlenbeck(geom):

    vtx = geom['coords']
    n = geom['n']
    l = 2
    C1 = torch.zeros(n,n)
    
    for i in range(n):
        Rsq = torch.sqrt(torch.pow((vtx[:,1] - vtx[i,1]),2) + torch.pow(vtx[:,2] - vtx[i,2],2))
        C1[i,:] = torch.exp(-Rsq/l);
    
    
    
    
    iC1 = torch.linalg.inv(C1)
    L1 = torch.cholesky(iC1,upper=True)

    return L1

def Draw_offset(intp_train_set,intp_test_set,samples_train,samples_test,freqs,seg_train,seg_test):
    
    multiplier_file = "bkg_multipliers/bkg_multipliers_samples_"+str(samples_train)+".npy"

    mult_tot = torch.zeros(samples_train,freqs,4)

    if os.path.isfile(multiplier_file):
        mult_tot = np.load(multiplier_file)        
        for i in range(samples_train):
            for j in range(freqs):
                segments_train = seg_train[i,j].bool()
                intp_train_set[4,i,j,segments_train] *= mult_tot[i,j,0]
                intp_train_set[5,i,j,segments_train] *= mult_tot[i,j,1]

                if i < samples_test:
                    segments_test = seg_test[i,j].bool()

                    intp_test_set[4,i,j,segments_test] *= mult_tot[i,j,2]
                    intp_test_set[5,i,j,segments_test] *= mult_tot[i,j,3]
                    
                    
    else:
        for i in range(samples_train):
            mult = 0.80+torch.rand(4)*0.40
            for j in range(freqs):
                segments_train = seg_train[i,j].bool()

                mult_freq = 0.95 + torch.rand(4)*0.1
                mult_freq[1] = 1
                mult_freq[3] = 1
                intp_train_set[4,i,j,segments_train] *= mult[0]*mult_freq[0]
                intp_train_set[5,i,j,segments_train] *= mult[1]*mult_freq[1]

                if i < samples_test:
                    segments_test = seg_test[i,j].bool()

                    intp_test_set[4,i,j,segments_test] *= mult[2]*mult_freq[2]
                    intp_test_set[5,i,j,segments_test] *= mult[3]*mult_freq[3]
                mult_tot[i,j] = mult*mult_freq 

        np.save(multiplier_file,mult_tot)


    return intp_train_set, intp_test_set



def Det_freq_dependency(samples,freqs,coeffs,segments,b):
    
    C = torch.zeros(samples,freqs,freqs)
    
    for s in range(samples):
        nodes = torch.sum(segments[s])
        init_scat = torch.sum(torch.mul(segments[s],coeffs[s,0]))/nodes
        c = (10*init_scat)/torch.exp(torch.tensor(-b*700))
        
        for f in range(freqs):
            new_scat_1 = c*torch.exp(torch.tensor(-b*(700+f*40)))
            for q in range(freqs):
                new_scat_2 = c*torch.exp(torch.tensor(-b*(700+q*40)))
                C[s,f,q] = new_scat_2/new_scat_1


    return C



def Det_norm_factor(qvec_scaling_file,samples,freqs,trainSet,device,mus_weight,geom,data,qvec):

    if os.path.isfile(qvec_scaling_file)==False:
        H_meas = torch.zeros(samples,freqs,device=device)
        H_DA = torch.zeros(samples,freqs,device=device)
        qvec_scaling = torch.zeros(samples,freqs,device=device)
        for p in range(samples):
            seg_temp = torch.nonzero(trainSet.segments[p,:]).to(device)
            seg_temp = seg_temp[:,0]
            for t in range(freqs):
                a = trainSet.true[0,p,t,:,None].to(device)
                b = trainSet.true[1,p,t,:,None].to(device)*mus_weight
                H = FE.Generate_H_torch(geom,a,b,qvec[:,[t]],device)
                
                temp = torch.mul(torch.exp(data[p,[t]]),1/a[:,0])    
                H_meas[p,t] = torch.sum(temp[0,:])
                flu_DA = torch.mul(H[0],1/a[:,0])
                H_DA[p,t] = torch.sum(flu_DA)
   
                qvec_scaling[p,t] = H_meas[p,t]/H_DA[p,t]

        qvec_scaling = torch.mean(qvec_scaling,dim=0)
        torch.save(qvec_scaling.cpu(),qvec_scaling_file) 


    else:
         qvec_scaling = torch.load(qvec_scaling_file) 
         
    return qvec_scaling


def Tranfer_to_device(dataSet,device):
    samples = len(dataSet.data) 
    current_input = dataSet.initial.float().to(device)
    C = dataSet.C.float().to(device)
    priors = dataSet.prior_vals.float().to(device)
    data = dataSet.data.float().to(device)
    segments = dataSet.segments.int().to(device)



    return samples,current_input,C,priors,data,segments


def Evaluate_set_EtoE(dataSet,xsize,ysize,samples,model,geom,cropped_indices,current_input,sample_info,outputs_eval,mus_weight,device):
       
    
    loss_total = 0
    freqs = len(current_input[0,0,:,0])
    
    model.eval()
    for i in range(samples):
        for j in range(0,freqs,2):
                                       
                
            optical_values = torch.cat((current_input[0,i,[j]],current_input[1,i,[j]],current_input[0,i,[j+1]]),dim=0).to(device)
            single_info = sample_info.Get_sample_info(i,j,device)

            temp_eval,loss = model(geom,optical_values,single_info,mus_weight,device=device,cropped_indices=cropped_indices)


            outputs_eval[:,i,j,cropped_indices[:,0]] = temp_eval[0:2]
            outputs_eval[:,i,j+1,cropped_indices[:,0]] = temp_eval[2:4]
                
                
                
            loss_total += loss

    #loss_total = loss_total/(freqs*samples)

    return outputs_eval,loss_total
    

def Evaluate_set_greedy(dataSet,model,device,samples,freqs,cropped_indices,outputs_eval,current_input,true_images_eval,C_wl,mus_weight,dx):

    loss_total = 0
    #single_data = torch.zeros(1,2,n)

    for i in range(samples):
        single_segments = dataSet.segments[i].to(device)

        for j in range(0,len(freqs),2):
            
            
            #if geom['data_assisted'] == True:
            #    single_data[0,:,cropped_indices[:,0]] = torch.exp(dataSet.data[None,i,j:j+2]).float()
            #    single_data_grid = torch.transpose(torch.reshape(single_data,(1,2,ysize,xsize)),2,3).to(device)
            #
            #else:
            single_data_grid = 0
                
            cur_freq = freqs[j]
            
            single_f = torch.cat((current_input[0,i,[cur_freq]],current_input[1,i,[cur_freq]],current_input[0,i,[cur_freq+1]]),dim=0)
            C= C_wl[i,cur_freq,cur_freq+1]
            bkgs = mus_weight

            single_dx = dx[:,i,int(cur_freq/2)]
            single_true = torch.cat((true_images_eval[:,i,cur_freq],true_images_eval[:,i,cur_freq+1]),dim=0).to(device)

            temp_eval,loss = model(single_f,single_dx,single_segments,single_true,bkgs,
                                   C,device=device,cropped_indices=cropped_indices,cur_data=single_data_grid)

            outputs_eval[:,i,cur_freq,cropped_indices[:,0]] = temp_eval[0:2]
            outputs_eval[:,i,cur_freq+1,cropped_indices[:,0]] = temp_eval[2:4]

            loss_total += loss
            
        #loss_total = loss_total/(len(freqs)*samples)

    return outputs_eval,loss_total


def Evaluate_set_Unet(dataSet,samples,freq_nro,outputs_eval,device,xsize,ysize,model_mus,model_mua,loss):
    
    loss_total = 0
    for i in range(samples):
        for j in range(freq_nro):
            single_data = dataSet.data[i,j].to(device)
            single_true = dataSet.true[0,[i],j].to(device)
            single_true_mus = dataSet.true[1,[i],j].to(device)

            single_segments = dataSet.segments[i].to(device)
            segs_inv = (single_segments == 0).nonzero().to(device)

            single_data_grid = torch.transpose(single_data.view(1,ysize,xsize),1,2)
            segs = torch.nonzero(single_segments)
            
            temp_out,loss = model_mua(single_data_grid[:,None,:,:],loss,single_true,segs,segs_inv)
            temp_out_mus,loss_mus = model_mus(single_data_grid[:,None,:,:],loss,single_true_mus,segs,segs_inv)
                
            outputs_eval[0,i,j] = temp_out[0]
            outputs_eval[1,i,j] = temp_out_mus[0]

            loss_total += loss + loss_mus

    #loss_total = loss_total/(len(freq_nro)*samples)


    return outputs_eval,loss_total
