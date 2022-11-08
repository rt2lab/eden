import torch
import torch.nn as nn
import numpy as np
    

def loss_survival(outputs, batch_ys, loss1_weight = 10, loss2_weight = 1, loss3_weight = 1, loss4_weight=1):
    
    ### Loss part 1 : mean square error between sequences weighted by proportion of zeros ####
    criterion_loss1 =  nn.BCELoss(reduction = 'none')
    if(batch_ys.sum() > 0):
        loss1 = criterion_loss1(outputs.double(),batch_ys)
        coeff_loss_1 = batch_ys.numel()/batch_ys.sum()
        coeff_loss_2 = batch_ys.numel()/(batch_ys.numel()-batch_ys.sum())
        range_coeff = coeff_loss_1+coeff_loss_2
        coeff_loss_1 = coeff_loss_1/range_coeff
        coeff_loss_2 = coeff_loss_2/range_coeff
        loss1 = loss1 * (coeff_loss_1*batch_ys+coeff_loss_2*(1-batch_ys))
    else:
        loss1 = criterion_loss1(outputs.double(),batch_ys)
        coeff_loss_1 = 0
        coeff_loss_2 = 1
        range_coeff = coeff_loss_1+coeff_loss_2
        coeff_loss_1 = coeff_loss_1/range_coeff
        coeff_loss_2 = coeff_loss_2/range_coeff
        loss1 = loss1 * (coeff_loss_1*batch_ys+coeff_loss_2*(1-batch_ys))
    loss1 = loss1_weight*loss1.mean()
    print("Loss1", loss1.detach().numpy())
        
    ### Loss part 2 : for patients who experienced the event ####
    #### High proba day of relapse ####
      
    output_size = batch_ys.shape[2]
    diff_y = batch_ys-torch.cat((torch.zeros(batch_ys.shape[0],1,output_size).double(),batch_ys[:,:-1,:]),dim = 1)
    relapse, mask_max_indices = torch.max(diff_y,dim = 1)
    
    if(relapse.sum()>0):
        loss2 = relapse*(1.-torch.gather(outputs,index = mask_max_indices.unsqueeze(dim = 1), dim = 1).squeeze(dim=1))
        loss2 = loss2_weight*torch.square(loss2).sum()/relapse.sum()
        print("Loss2", loss2.detach().numpy())
    else:
        loss2 = torch.zeros(1)
    
    ### Loss part 3 : for patients who experienced the event ####
    #### Low proba before #####
    if(relapse.sum()>0):
        indexes2 =  (relapse*(mask_max_indices-1)).to(dtype = torch.int64)
        loss3 = relapse*(torch.gather(outputs,index = indexes2.unsqueeze(dim=1), dim = 1).squeeze(dim=1))
        loss3 = loss3_weight * torch.square(loss3).sum()/relapse.sum()
        print("Loss3", loss3.detach().numpy())
    else:
        loss3=torch.zeros(1)
   
    
    if (len(relapse)-relapse.sum()) >0:
        loss4 = (1-relapse)*(outputs[:,-1,:])
        loss4 = loss4_weight * torch.square(loss4).sum()/(relapse.numel()-relapse.sum())
        print("Loss4", loss4.detach().numpy())
    else:
        loss4 = torch.zeros(1)

    loss = loss1 + loss2 + loss3 + loss4
    return(loss, np.array([loss1.detach().numpy(),loss2.detach().numpy(),loss3.detach().numpy(), loss4.detach().numpy()]))


def loss_bce(outputs,batch_ys, loss_weight1 = None, loss_weight2 = None,loss_weight3 = None,loss_weight4 = None ):
    
    criterion =  nn.BCELoss(reduction = 'none')
    
    if(batch_ys.sum() > 0):
        loss = criterion(outputs.double(),batch_ys)
        coeff_loss_1 = batch_ys.numel()/batch_ys.sum()
        coeff_loss_2 = batch_ys.numel()/(batch_ys.numel()-batch_ys.sum())
        range_coeff = coeff_loss_1+coeff_loss_2
        coeff_loss_1 = coeff_loss_1/range_coeff
        coeff_loss_2 = coeff_loss_2/range_coeff
        loss = loss * (coeff_loss_1*batch_ys+coeff_loss_2*(1-batch_ys))
    else:
        loss = criterion(outputs.double(),batch_ys)
        coeff_loss_1 = 0
        coeff_loss_2 = 1
        range_coeff = coeff_loss_1+coeff_loss_2
        coeff_loss_1 = coeff_loss_1/range_coeff
        coeff_loss_2 = coeff_loss_2/range_coeff
        loss = loss * (coeff_loss_1*batch_ys+coeff_loss_2*(1-batch_ys))
      
    return(loss.mean(), np.array([0,0,0,0]))
