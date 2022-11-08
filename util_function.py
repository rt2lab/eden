import numpy as np
import torch
import pickle
#-*- coding: utf-8 -*-

#Loading function 
def load_pkl(path):
    with open(path):
        obj = pickle.load(open( path , "rb" ) )
        return obj


#Function to loop over all batches, called by train_model
def loop_over_all_batches(lstm,
                          data_train_batches,labels_train_batches,elapsed_train_batches,
                          optimizer, criterion, scheduler = None,  previous_loss = None,
                          data_val_batches = None ,labels_val_batches = None ,elapsed_val_batches = None,
                          compute_train_loss = True, compute_train_pred = True,
                          compute_val_loss = False, compute_val_pred = False,
                          compute_optimizer = True,
                          loss1_weight = 10,loss2_weight = 1,
                          loss3_weight = 1,loss4_weight=1):

    total_cost_train, total_cost_val = 0, 0
    Y_true, Y_pred, Y_true_val, Y_pred_val = [], [], [], []
    loss_tot_train =[0]*4
    loss_tot_val = [0]*4

    #If optimizing on the train should be done
    if(compute_train_loss or compute_train_pred or compute_optimizer):

        number_batches_train = len(data_train_batches)
        print("Number batches train: ", number_batches_train)
        
        total_cost_train = 0
        
        #Loop over all train batches
        for i in range(number_batches_train):
            print("Batch", i)
            batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], elapsed_train_batches[i]

            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
            batch_ys = torch.from_numpy(batch_ys)
            
            if(compute_optimizer):
                
                optimizer.zero_grad()
                outputs = lstm(batch_ts, batch_xs)
                loss, loss_tot = criterion(outputs, batch_ys, loss1_weight = loss1_weight,loss2_weight = loss2_weight,loss3_weight = loss3_weight,loss4_weight=loss4_weight)
                print("Loss: " ,loss)
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()
                
                if(scheduler):
                    scheduler.step()

            if( not compute_optimizer and (compute_train_loss or compute_train_pred)):
                outputs = lstm(batch_ts, batch_xs)
                
            if(compute_train_loss):
                total_cost_train += loss.detach().numpy()
                loss_tot_train = loss_tot_train + loss_tot

            if(compute_train_pred):
                Y_true.append(batch_ys.detach().numpy())
                Y_pred.append(outputs.detach().numpy())
                
            del loss, outputs #To keep memory constant
        
        print(total_cost_train)
        total_cost_train = total_cost_train / number_batches_train
        print(loss_tot_train)
        print(number_batches_train)
        loss_tot_train = loss_tot_train / number_batches_train

    #Loop on validation batches
    if(compute_val_loss or compute_val_pred):
        
        number_batches_val = len(data_val_batches)
        print("Number batches val: ", number_batches_val)
        for i in range(number_batches_val):
            
            batch_xs, batch_ys, batch_ts = data_val_batches[i], labels_val_batches[i], elapsed_val_batches[i]
            
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
            batch_ys = torch.from_numpy(batch_ys)

            outputs = lstm(batch_ts, batch_xs)        

            if(compute_val_loss):
                
                total_loss, loss_tot = criterion(outputs, batch_ys, loss1_weight = loss1_weight,loss2_weight = loss2_weight,loss3_weight = loss3_weight,loss4_weight=loss4_weight)
                total_cost_val += total_loss.detach().numpy()
                loss_tot_val = loss_tot_val + loss_tot

            if(compute_val_pred):
                Y_true_val.append(batch_ys.detach().numpy())
                Y_pred_val.append(outputs.detach().numpy()) 

        total_cost_val = total_cost_val / number_batches_val
        loss_tot_val = loss_tot_val / number_batches_val

    return(previous_loss, total_cost_train,total_cost_val,Y_true,Y_pred,Y_true_val,Y_pred_val, loss_tot_train, loss_tot_val)
