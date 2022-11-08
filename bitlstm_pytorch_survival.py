#############################################################################
#Step 1 : import modules
#############################################################################

import torch
import torch.nn as nn
import numpy as np

#############################################################################
#Step 2 : defining discounting functions
#############################################################################

def g_inv(x):
    eps = 1e-16
    return(1/(x+eps))

def g_log(x):
    log_x = torch.log(torch.add(x,np.e))
    return(1/log_x)

def g_const(x):
    return(torch.ones(x.size()))
   
#############################################################################
#Step 3 : define the class
#############################################################################

class biTLSTM(nn.Module): #the class inherits from nn.Module
    
    def __init__(self,input_sz : int, hidden_sz : int, fc_sz :int, output_sz : int, dropout_rate : float,
                 discount = "log", bidirectional = False, survival = False, embed_sz = 28):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.output_size = output_sz
        
        #Embedding
        self.embed = nn.Parameter(torch.Tensor(input_sz,embed_sz))
        
        #Concatenation matrices for bi-LSTM
        self.W_forward = nn.Parameter(torch.Tensor(embed_sz,hidden_sz * 4))
        self.U_forward = nn.Parameter(torch.Tensor(hidden_sz,hidden_sz * 5))
        self.bias_forward = nn.Parameter(torch.Tensor(hidden_sz * 5))
        if(bidirectional):
            self.W_backward = nn.Parameter(torch.Tensor(embed_sz,hidden_sz * 4))
            self.U_backward = nn.Parameter(torch.Tensor(hidden_sz,hidden_sz * 5))
            self.bias_backward = nn.Parameter(torch.Tensor(hidden_sz * 5))
        
        #Fully connected layer for the output 
        if(bidirectional):
            self.fc = nn.Linear(2*hidden_sz, fc_sz)
        else:
            self.fc = nn.Linear(hidden_sz, fc_sz)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_rate)
        self.out = nn.Linear(fc_sz, output_sz)

        #Discount function for time
        if(discount == "log"):
            self.g = g_log
        elif (discount == "inv"):
            self.g = g_inv
        else:
            self.g = g_const
            
        #Architecture
        self.bidirectional = bidirectional
        self.survival = survival
        
    #Weight initialization
    def init_weights(self): 
        nn.init.xavier_normal_(self.embed,gain = nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.W_forward,gain = nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.U_forward,gain = nn.init.calculate_gain("relu"))
        nn.init.normal_(self.bias_forward)
        if(self.bidirectional):
            nn.init.xavier_normal_(self.W_backward,gain = nn.init.calculate_gain("relu"))
            nn.init.xavier_normal_(self.U_backward,gain = nn.init.calculate_gain("relu"))
            nn.init.normal_(self.bias_backward)
         
    def forward(self, delta, batch_xs, init_states = None):    
        
        batch_xs = torch.from_numpy(batch_xs).float()
        batch_sz, seq_sz, _ = batch_xs.size()
        delta = torch.from_numpy(delta).float().unsqueeze(2)
        delta_g = self.g(delta)
        
        if(init_states is None):
            h_t,c_t,h2_t,c2_t = ( torch.zeros(batch_sz,self.hidden_size),
                        torch.zeros(batch_sz,self.hidden_size),
                        torch.zeros(batch_sz,self.hidden_size),
                        torch.zeros(batch_sz,self.hidden_size))
        else:
            h_t, c_t = init_states
        
        HS = self.hidden_size
        
        #Embedding layer
        v = batch_xs@self.embed
        
        #Foward LSTM
        ht_forward_seq = torch.zeros(batch_xs.shape[0],0,self.hidden_size)
        for t in range(seq_sz):
            
            x_t = v[:,t,:]
            delta_t = delta_g[:,t,:].squeeze().reshape(-1,1)   
            
            gates = x_t @ self.W_forward + h_t @ self.U_forward[:,:4*HS] + self.bias_forward[:4*HS]
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),
                torch.sigmoid(gates[:, HS:HS*2]),
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:])  
            )
            
            short_term_memory = torch.tanh(c_t @ self.U_forward[:,4*HS:5*HS] + self.bias_forward[4*HS:5*HS])
            discounted_short_term = short_term_memory * delta_t.expand_as(short_term_memory) 
            long_term_memory = c_t - short_term_memory
            adjusted_previous_memory = long_term_memory + discounted_short_term
            
            c_t = f_t * adjusted_previous_memory + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            ht_forward_seq = torch.cat((ht_forward_seq,h_t.unsqueeze(dim=1)), dim = 1)
        
        #Backward LSTM
        if(self.bidirectional):
            ht_backward_seq = torch.zeros(batch_xs.shape[0],0,self.hidden_size)
            for t in range(seq_sz-1,-1,-1):
                #print(t)
                
                x_t = v[:,t,:]
                delta_t = delta_g[:,t,:].squeeze().reshape(-1,1)
                
                gates = x_t @ self.W_backward + h2_t @ self.U_backward[:,:4*HS] + self.bias_backward[:4*HS]
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),
                    torch.sigmoid(gates[:, HS:HS*2]),
                    torch.tanh(gates[:, HS*2:HS*3]),
                    torch.sigmoid(gates[:, HS*3:])  
                )
                
                short_term_memory = torch.tanh(c2_t @ self.U_backward[:,4*HS:5*HS] + self.bias_backward[4*HS:5*HS])
                discounted_short_term = short_term_memory * delta_t.expand_as(short_term_memory) 
                long_term_memory = c2_t - short_term_memory
                adjusted_previous_memory = long_term_memory + discounted_short_term
                
                c2_t = f_t * adjusted_previous_memory + i_t * g_t
                h2_t = o_t * torch.tanh(c2_t)
                ht_backward_seq = torch.cat((ht_backward_seq,h2_t.unsqueeze(dim=1)), dim = 1)           
            
            #Concatenate forward and backward passes
            ht_backward_seq = ht_backward_seq.flip(dims=[1])
            h_t = torch.cat((ht_forward_seq,ht_backward_seq), dim = 2)
        else:
            h_t = ht_forward_seq
            
        #Fully connected output layer
        fc_out = self.relu(self.fc(h_t))
        fc_out = self.dropout(fc_out)
        out_t = torch.sigmoid(self.out(fc_out))

        #Survival form of the output : passing from h to W using chain rule
        if self.survival:
            out = torch.ones(batch_xs.shape[0],self.output_size)
            out_seq = torch.zeros(batch_xs.shape[0],0, self.output_size)
            for t in range(seq_sz):
                out = out * (1-out_t[:,t,:])
                out_seq = torch.cat((out_seq,1-out.unsqueeze(1)), dim = 1)
        else:
            out_seq = out_t
                
        return out_seq