def run_model_on_test(path_results, model = "TLSTM",hidden_sz = 1024 ,fc_sz = 1024,
                  dropout_rate = 0.5, discount = "log",learning_rate = 1e-3,
                  embed_sz = 25,attention_medical_code = True, attention_medical_size = 25,
                  survival = True, bidirectional = False):
    
    ###############################################################################
    #Load libraries and util functions from file
    ###############################################################################
       
        #General python requirements
    import torch
    import pandas as pd
    import numpy as np
    from util_function import load_pkl
    
        #Model defintion 
    from bitlstm_pytorch_survival import biTLSTM
        
    ###############################################################################
    ##########################  Load test data ####################################
    ###############################################################################
    
    data_test = load_pkl(path_results + '/data_test' + '.p')
    elapsed_test = load_pkl(path_results + '/elapsed_test' + '.p')
    label_test = load_pkl(path_results + '/labels_test' + '.p')
    
    ###############################################################################
    ##########################  Data dimension ####################################
    ###############################################################################    
    
    input_sz = data_test[0].shape[-1]
    print("Total input size", input_sz)
    output_sz = label_test[0].shape[-1]
    print("Output_dim : ",output_sz)
    
    ###############################################################################
    ########################## Load saved model ###################################
    ###############################################################################

    lstm = biTLSTM(input_sz, hidden_sz, fc_sz, output_sz, dropout_rate, discount, bidirectional, survival, embed_sz)    
    lstm.load_state_dict(torch.load(path_results + "/data/model"))    
    
    ###############################################################################
    ########################## Run model ##########################################
    ###############################################################################
    
    Y_true_test,Y_pred_test = [],[]
    number_batches_test = len(data_test)
    print("Number batches test: ", number_batches_test)
    for i in range(number_batches_test):
        
        batch_xs, batch_ys, batch_ts = data_test[i], label_test[i], elapsed_test[i]
        
        batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
        batch_ys = torch.from_numpy(batch_ys)
    
        outputs = lstm(batch_ts, batch_xs)        
    
        Y_true_test.append(batch_ys.detach().numpy())
        Y_pred_test.append(outputs.detach().numpy()) 
    
    ###############################################################################
    ########################## Save results #######################################
    ###############################################################################
    
    #Save a more compact form of the total logit prediction for the last epoch    
    Y_true_concat = np.concatenate([Y_true_test[i].reshape(-1,output_sz) for i in range(len(Y_true_test))], axis=0)
    Y_logit_concat = np.concatenate([Y_pred_test[i].reshape(-1,output_sz) for i in range(len(Y_pred_test))], axis=0)   
                    
    Y_concat = np.concatenate((Y_true_concat,Y_logit_concat), axis =1)
    out_df = pd.DataFrame(Y_concat)
    out_df.to_csv(path_results + "/data/results_test_concatenate.csv")
    
    return("All good")
