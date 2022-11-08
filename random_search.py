###############################################################################
#############   Loading packages and set parameters ##################
###############################################################################

###############################################################################
#Pathways 
import numpy as np
import pandas as pd
from train_model import train_model

#Constant parameters
learning_rate = 1e-3
training_epochs = 1

# Hyperparameters ranges
embed_sz_range = [25,50,128,256]
hidden_sz_range = [128,256,512,1024,2048]
fc_sz_range = [128,256,512,1024,2048]
dropout_rate_range = [0,0.25,0.5,0.75]

#Nb repetitions
nb_repeat = 50
path_results = "xx" #Your path to the results folder here. Should contain a directory data.

###############################################################################
######################   Random search ########################################
###############################################################################

tested_embed_sz, tested_hidden_sz, tested_fc_sz, tested_dropout_rate, val_losses = [],[],[],[],[]
for n in range(nb_repeat):
    
    print("Repetition ",n+1 , "/",nb_repeat)
    
    #Random selection of hyperparameters
    embed_sz = np.random.choice(embed_sz_range)
    hidden_sz = np.random.choice(hidden_sz_range)
    fc_sz = np.random.choice(fc_sz_range)
    dropout_rate = np.random.choice(dropout_rate_range)
    
    #Save paramters
    tested_embed_sz.append(embed_sz)
    tested_hidden_sz.append(hidden_sz)
    tested_fc_sz.append(fc_sz)
    tested_dropout_rate.append(dropout_rate)
    print(f"Embed_size : {embed_sz}; hidden size: {hidden_sz}; fc_sz : {fc_sz}; dropout rate : {dropout_rate}")
    
    #Run model
    val_loss = train_model(path_results,
              model = "TLSTM",
              hidden_sz = hidden_sz ,fc_sz = fc_sz,
              dropout_rate = dropout_rate, discount = "log",
              learning_rate = learning_rate,
              training_epochs = training_epochs,
              embed_sz = embed_sz,
              survival = True, bidirectional = True, save_results = False, return_val_loss = True)
    
    #Save loss on validation set
    val_losses.append(val_loss)
    
###############################################################################
######################   Save results #########################################
###############################################################################    

#Print best results
i = np.argmin(val_losses)
print("Best loss val is :", val_losses[i], " for embed size ",tested_embed_sz[i],
      ", hidden size ", tested_hidden_sz[i], 
      " ,fc size ", tested_fc_sz[i],
      " ,dropout rate ", tested_dropout_rate[i])

#Save results
Y_concat = np.concatenate((np.expand_dims(tested_embed_sz,axis=1),
                           np.expand_dims(tested_hidden_sz,axis=1),
                          np.expand_dims(tested_fc_sz,axis=1),
                          np.expand_dims(tested_dropout_rate,axis=1),
                           np.expand_dims(val_losses,axis=1)), axis =1)
out_df = pd.DataFrame(Y_concat,columns=["Embed_sz","Hidden_sz","Fc_sz","dropout_rate","val_losses"])
out_df.to_csv(path_results + "/data/random_search.csv")
