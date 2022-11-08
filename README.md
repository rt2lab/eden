EDEN (Event DEtection Network) is a non-parametric bi-directional LSTM network which aims at detecting survival events in longitudinal, irregularly time-spaced and right-censored medical sequences. Given patient information during medical visits, it can identify, date and classify several survival endpoints.

Description of the code

This repository contains the full framework to train and test EDEN. We detail below how you should run the code: 

Step 1 : create a repository that will contain your data and your results.
This repository should contain a directory call data; which should itself contain all of your data, namely: 

-data_train.npy : the sequence of medical codes retrieved in medical visits for train data.Should be a list of size the number of mini-batches. 
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visits * vocabulary_size
-elapsed_train.npy : the sequence of time interval between medical visits for train data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * 1 
-labels_train.npy : Sequence of event rate function value (0 before relapse, 1 after relapses) for train data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * number_of_survival_endpoints

-data_val.npy : the sequence of medical codes retrieved in medical visits for validation data.Should be a list of size the number of mini-batches. 
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visits * vocabulary_size
-elapsed_val.npy : the sequence of time interval between medical visits for validation data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * 1 
-labels_val.npy : Sequence of event rate function value (0 before relapse, 1 after relapses) for validation data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * number_of_survival_endpoints

-data_test.npy : the sequence of medical codes retrieved in medical visits for test data.Should be a list of size the number of mini-batches. 
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visits * vocabulary_size
-elapsed_test.npy : the sequence of time interval between medical visits for test data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * 1 
-labels_test.npy : Sequence of event rate function value (0 before relapse, 1 after relapses) for test data. Should be a list of length the number of batches
Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * number_of_survival_endpoints

Step 2 : Random search for hyperparameters

By running random_seach.py you can proceed to random search for the set of hyperparameters of your choice.
The ranges of the hyperparameters, along with the number of repetitions should be define at the beginning of the file.
Outputs : a csv file random_search.csv contaning the list of searched hyperparameters and the loss on the validation (last epoch only) obtained for this set of parameters.

Step 3 : Training of the model

By running run_model.py you can train the model on the train set, save the model and get results on the test set.
Hyperparameters values should be entered at the beginning of the file.
The code then :
-train the model on the train set
-save the loss functions value for each epoch for train and validation set
-save the predictions for the validation set after the last epoch 
-save the model after the last epoch
-run the model on the test set and save the predictions

Ouputs : 
-model.py : the trained model
-total_cost_train : a csv file containing the values of the total loss function for each epoch on the training set 
-total_cost_val : a csv file containing the values of the total loss function for each epoch on the validation set 
-total_cost_train : a csv file containing the values of each component of the loss function (l1,l2,l3, and l4) for each epoch on the training set 
-total_cost_val : a csv file containing the values of each component of the loss function (l1,l2,l3, and l4) for each epoch on the validation set 
-results_concatenate : a csv_file containing the prediction and the true values of the event rate function W for each survival endpoint for the validation set after the last epoch.
-results_test_concatenate : a csv_file containing the prediction and the true values of the event rate function W for each survival endpoint for the test set after the last epoch. 
