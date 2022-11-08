##############################################################################
# Define parameters
##############################################################################

from get_test_results import run_model_on_test
from train_model import train_model
hidden_sz = 10  # Hidden size
fc_sz = 20  # Fully connected size
embed_sz = 50  # Embedding size
dropout_rate = 0.5  # Dropout rate
learning_rate = 1e-3  # Learning rate
training_epochs = 1  # Number of epochs
bidirectional = True  # Should the T-LSTM be directional?
survival = True  # Should the model use survival output form and survival function?

##############################################################################
# Path
##############################################################################

# Path were to put the results. Should contain a directory data with the data inside
path_results = "xx"

##############################################################################
# Load util functions
##############################################################################

##############################################################################
# Training
##############################################################################

train_model(path_results,
            model="TLSTM",
            hidden_sz=hidden_sz,
            fc_sz=fc_sz,
            dropout_rate=dropout_rate,
            discount="log",
            learning_rate=learning_rate,
            training_epochs=training_epochs,
            embed_sz=embed_sz,
            survival=True,
            bidirectional=True)

##############################################################################
# Test
##############################################################################
run_model_on_test(path_results,
                  model="TLSTM",
                  hidden_sz=hidden_sz,
                  fc_sz=fc_sz,
                  dropout_rate=dropout_rate,
                  discount="log",
                  learning_rate=learning_rate,
                  embed_sz=embed_sz,
                  l=True,
                  bidirectional=True)
