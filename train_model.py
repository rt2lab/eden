def train_model(path_results, model="TLSTM", hidden_sz=1024, fc_sz=1024,
                dropout_rate=0.5, discount="log", learning_rate=1e-3,
                training_epochs=100, embed_sz=25,
                attention_medical_code=True, attention_medical_size=25,
                survival=True, bidirectional=False,
                save_results=True, return_val_loss=False,
                loss1_weight=10, loss2_weight=1, loss3_weight=1, loss4_weight=1):

    ###############################################################################
    # Loading dependencies
    ###############################################################################
    # Load libraries and util functions from file

    # General python requirements
    import numpy as np
    import pandas as pd

    # Pytorch
    import torch

    # Model defintion
    from bitlstm_pytorch_survival import biTLSTM

    # Util functions
    from util_function import loop_over_all_batches

    # Loss functions
    from loss_functions import loss_survival
    from loss_functions import loss_bce

    ###############################################################################
    ######  Separation of training and validation #################################
    ###############################################################################

    # Sequence of medical codes retrieved in medical visits.
    # Should be a list of length the number of batches
    # Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * vocabulary_size
    data_train_batches = np.load(
        path_results+"/data/data_train" + ".npy",
        allow_pickle=True
    )
    data_val_batches = np.load(
        path_results+"/data/data_val" + ".npy",
        allow_pickle=True
    )

    # Sequence of time interval between medical visit
    # Should be a list of length the number of batches
    # Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * 1
    elapsed_train_batches = np.load(
        path_results+"/data/elapsed_train" + ".npy",
        allow_pickle=True
    )
    elapsed_val_batches = np.load(
        path_results+"/data/elapsed_val" + ".npy",
        allow_pickle=True
    )

    # Sequence of event rate function value (0 before relapse, 1 after relapses)
    # Should be a list of length the number of batches
    # Each element of the list (mini-batch) should be of size nb_patient * nb_medical_visit * number_of_events
    labels_train_batches = np.load(
        path_results+"/data/labels_train" + ".npy",
        allow_pickle=True
    )
    labels_val_batches = np.load(
        path_results+"/data/labels_val" + ".npy",
        allow_pickle=True
    )

    ###############################################################################
    ######################   Data dimension #######################################
    ###############################################################################

    input_sz = data_train_batches[1].shape[-1]
    print("Total input size", input_sz)

    output_sz = labels_train_batches[0].shape[-1]
    print("Output_dim : ", output_sz)

    ###############################################################################
    ##################### Model creation  #########################################
    ###############################################################################

    # Create model, init weights, define loss function and optimizer
    lstm = biTLSTM(input_sz, hidden_sz, fc_sz, output_sz,
                   dropout_rate, discount, bidirectional,
                   survival, embed_sz)
    lstm.init_weights()

    if(survival):
        criterion = loss_survival
    else:
        criterion = loss_bce
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    scheduler = None

    ###############################################################################
    ##################### Model training  #########################################
    ###############################################################################

    # We stored results for train and val (cost and avg precision)
    total_cost_train, total_cost_val, loss_tot_train, loss_tot_val = [], [], [], []
    previous_loss = None

    # Loop over epochs
    for epoch in range(training_epochs):

        print("Epoch n" + str(epoch+1) + "/" + str(training_epochs))

        # Loop over all batches for train
        (previous_loss, cost_traink, _, Y_truek,
         Y_predk, _, _, loss_tot_traink, _) = loop_over_all_batches(
            lstm,
            data_train_batches, labels_train_batches,
            elapsed_train_batches, optimizer, criterion,
            scheduler, previous_loss,
            None, None, None, compute_train_loss=True, compute_train_pred=True,
            compute_val_loss=False, compute_val_pred=False,
            loss1_weight=loss1_weight, loss2_weight=loss2_weight,
            loss3_weight=loss3_weight, loss4_weight=loss4_weight
        )

        # Save cost for epoch
        total_cost_train.append(cost_traink)
        print("Cost train:", cost_traink)
        loss_tot_train.append(loss_tot_traink)

        # Results on validation data
        (_, _, cost_valk, _, _, Y_true_valk,
         Y_pred_valk, _, loss_tot_valk) = loop_over_all_batches(
            lstm,
            None, None, None, None, criterion, scheduler, previous_loss,
            data_val_batches, labels_val_batches, elapsed_val_batches,
            compute_val_loss=True,
            compute_val_pred=True,
            compute_train_loss=False,
            compute_train_pred=False,
            compute_optimizer=False,
            loss1_weight=loss1_weight, loss2_weight=loss2_weight,
            loss3_weight=loss3_weight, loss4_weight=loss4_weight
        )

        # scheduler.step(cost_valk)
        total_cost_val.append(cost_valk)
        print("Cost val:", cost_valk)
        loss_tot_val.append(loss_tot_valk)

    ###############################################################################
    ##################### Save model and results  #################################
    ###############################################################################

    if(save_results):
        # Save torch model
        torch.save(lstm.state_dict(), path_results + "/data/model")

        # Save train/val loss and average precision
        arr_total_cost_train = np.column_stack(
            (np.arange(1, training_epochs+1), np.hstack(total_cost_train)))
        pd.DataFrame(arr_total_cost_train, columns=["epoch", "value"]).to_csv(
            path_results + "/data/total_cost_train.csv")

        arr_total_cost_val = np.column_stack(
            (np.arange(1, training_epochs+1), np.hstack(total_cost_val)))
        pd.DataFrame(arr_total_cost_val, columns=["epoch", "value"]).to_csv(
            path_results + "/data/total_cost_val.csv")

        # Save a more compact form of the total logit prediction for the last epoch
        Y_true_concat = np.concatenate(
            [Y_true_valk[i].reshape(-1, output_sz) for i in range(len(Y_true_valk))], axis=0)
        Y_logit_concat = np.concatenate(
            [Y_pred_valk[i].reshape(-1, output_sz) for i in range(len(Y_true_valk))], axis=0)

        # Total loss
        total_loss_train = np.concatenate(
            [loss_tot_train[i].reshape(-1, 4) for i in range(len(loss_tot_train))], axis=0)
        total_loss_val = np.concatenate(
            [loss_tot_val[i].reshape(-1, 4) for i in range(len(loss_tot_val))], axis=0)
        pd.DataFrame(total_loss_train, columns=["l1", "l2", "l3", "l4"]).to_csv(
            path_results + "/data/detail_cost_train.csv")
        pd.DataFrame(total_loss_val, columns=["l1", "l2", "l3", "l4"]).to_csv(
            path_results + "/data/detail_cost_val.csv")

        Y_concat = np.concatenate((Y_true_concat, Y_logit_concat), axis=1)
        out_df = pd.DataFrame(Y_concat)
        out_df.to_csv(path_results + "/data/results_concatenate.csv")

    if return_val_loss:
        return(total_cost_val[-1])

    return("All good, training is over!")
