"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import torch
import argparse
import datetime
import time
import numpy as np
import math
import pandas as pd
import util.dataset as ds
import util.utils as utils
import layer.TransformerGAT as tst

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import MinMaxScaler, StandardScaler


torch.manual_seed(42)
np.random.seed(42)


def train(model, training_time_data, src_mask, tgt_mask, loss_fn, optimizer, scheduler, batch_first, batch_size, input_size, LINEAR_DECODER=False):
    model.train() # Turn on the train mode \o/
    start_time = time.time()
    total_loss = 0.

    for step, batch in enumerate(training_time_data):
        optimizer.zero_grad()
        
        src, trg, trg_y = batch
        B, N, T, F = trg_y.size()
        if input_size == 1: # todo
            trg_y.unsqueeze(-1) # feature size = 1
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)

        # Split the data tensor along dimension 1 into smaller tensors
        src_chunks, trg_chunks, trg_y_chunks = src.chunk(src.size(1) // batch_size, dim=1), trg.chunk(trg.size(1) // batch_size, dim=1), trg_y.chunk(trg_y.size(1) // batch_size, dim=1)
        # Create a list of TensorDatasets
        minibatch_datasets = [TensorDataset(src_chunk, trg_chunk, trg_y_chunk) for src_chunk, trg_chunk, trg_y_chunk in zip(src_chunks, trg_chunks, trg_y_chunks)]

        # Create a list of DataLoaders
        minibatch_dataloaders = [DataLoader(minibatch_dataset, batch_size=batch_size, shuffle=False) for minibatch_dataset in minibatch_datasets]

        for minibatch_dataloader in minibatch_dataloaders:
            for minibatch_idx, (src, trg, trg_y) in enumerate(minibatch_dataloader):

                # Permute from shape [series batch size, node minibatch size, seq len, num features] to [seq len, series batch size*node minibatch size, num features]
                # Node dimension is put inside the batch, in order to process each node along the time separately
                if batch_first == False:
                    src = src.permute(2, 0, 1, 3)
                    src = src.reshape(src.size()[0], src.size()[1] * src.size()[2], src.size()[3])
                    # print("src shape changed from {} to {}".format(shape_before, src.shape))

                    trg = trg.permute(2, 0, 1, 3)
                    trg = trg.reshape(trg.size()[0], trg.size()[1] * trg.size()[2], trg.size()[3])

                    trg_y = trg_y.permute(2, 0, 1, 3)
                    trg_y = trg_y.reshape(trg_y.size()[0], trg_y.size()[1] * trg_y.size()[2], trg_y.size()[3])

                output = model(
                    src=src,
                    tgt=trg,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    linear_decoder=LINEAR_DECODER
                )
                # output = output.permute(1, 0, 2).squeeze()
                # print(f'output:', output.size())
                if batch_first == False:
                    output = output.permute(1, 0, 2)
                    trg_y = trg_y.permute(1, 0, 2)
            
                loss = loss_fn(output, trg_y) #output.flatten()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        ave_batch_loss = total_loss / B
        elapsed = time.time() - start_time
        print('| epoch {:3d} | step {:3d} | '
                'lr {:02.8f} | {:5.2f} ms | '
                'batch node loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, step, scheduler.get_last_lr()[0], # get_lr()
                elapsed * 1000, ave_batch_loss, math.exp(ave_batch_loss))) # math.log(cur_loss)
        
    total_loss = 0
    start_time = time.time()
    scheduler.step()

    return output

def expectile_loss(pred, target, expectile_level):
    """
    taking the maximum of two terms:(expectile_level - 1) * abs_errors and expectile_level * errors.
    """
    errors = target - pred
    abs_errors = torch.abs(errors)
    expectile_loss = torch.mean(torch.max((expectile_level - 1) * abs_errors, expectile_level * errors))
    return expectile_loss


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training"
    )
    argparser.add_argument("--data_file", type=str, default='forecast_cyclical_data',
                           help="file name wo/ suffix")
    argparser.add_argument("--projection_map_file", type=str, default='projection_map',
                           help="file name wo/ suffix")
    argparser.add_argument("--SCALER", type=bool, default=True)
    argparser.add_argument("--LINEAR_DECODER", type=bool, default=False)
    argparser.add_argument("--test_size", type=float, default=0.2)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--dim_val", type=int, default=512)
    argparser.add_argument("--n_heads", type=int, default=8)
    argparser.add_argument("--n_decoder_layers", type=int, default=4)
    argparser.add_argument("--n_encoder_layers", type=int, default=4)
    argparser.add_argument("--enc_seq_len", type=int, default=4,
                           help="length of input given to encoder 153")
    argparser.add_argument("--dec_seq_len", type=int, default=1,
                           help="length of input given to decoder 92")
    argparser.add_argument("--output_sequence_length", type=int, default=1,
                           help="target sequence length. If hourly data and length = 48, you predict 2 days ahead 48")
    argparser.add_argument("--step_size", type=int, default=1,
                           help="Step size, i.e. how many time steps does the moving window move at each step")
    argparser.add_argument("--in_features_encoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--in_features_decoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--batch_first", type=bool, default=False)
    argparser.add_argument("--target_col_name", type=str, default="Reliability")
    argparser.add_argument("--timestamp_col", type=str, default="Timestamp")
    argparser.add_argument("--node_col", type=str, default="Node")
    argparser.add_argument("--label_col", type=str, default="Node Label")
    argparser.add_argument("--exogenous_vars", type=str, default="Flexibility,Service,Infrastructure Quality,Freight",
                           help="split by comma, should contain strings. Each string must correspond to a column name")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    # Only use data from this date and onwards
    # cutoff_date = datetime.datetime(2017, 1, 1) 

    # Define input variables 
    if args.LINEAR_DECODER:
        args.output_sequence_length = args.enc_seq_len
    window_size = args.enc_seq_len + args.output_sequence_length # used to slice data into sub-sequences
    exogenous_vars = args.exogenous_vars.split(',')
    input_variables = [args.target_col_name] + exogenous_vars
    input_size = len(input_variables)

    # Read data
    # Input x
    # (batch_size, nodes, sequentials, features)
    data, slice_size = utils.read_data(file_name=args.data_file, node_col_name=args.node_col, timestamp_col_name=args.timestamp_col)

    # Remove test data from dataset for each node
    ratio = round(slice_size*(1-args.test_size))
    first_round = data.iloc[0:ratio, :]
    for i in range(1,round(len(data)//slice_size)+1):
        first_round = pd.concat([first_round, data.iloc[slice_size*i:slice_size*i+ratio, :]], axis=0)
    training_time_data = first_round
    training_slice_size = ratio

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
    # Should be training time series data indices only
    training_indices = utils.get_indices_entire_sequence(
        data=training_time_data, 
        window_size=window_size, 
        step_size=args.step_size,
        slice_size=training_slice_size)

    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    # Recover the original values
    # original_data = scaler.inverse_transform(scaled_data)
    map_series = training_time_data[input_variables].values
    labels = training_time_data[args.label_col].values
    
    # dic for label wise feature projection, e.g., OrderedDict([(0, 3), (1, 2))])
    dic = utils.read_projection_map(file_name=args.projection_map_file)
    series = np.zeros((len(map_series), sum(dic.values()))) # 0 avoid the impact of -1 values for the scaler func.
    # series = np.full((len(map_series), sum(dic.values())), -1.) # -1 denotes the absence feature of each node
    for i in range(len(series)):
        given_index = labels[i]
        index = utils.index_for_feature_projection(dic, given_index)
        series[i][index:index+dic[given_index]] = map_series[i][map_series[i] != -1]

    if args.SCALER:
        amplitude = scaler.fit_transform(series)
    else:
        amplitude = series
        
    # Making instance of custom dataset class
    training_time_data = ds.TransformerDataset(
        data=torch.tensor(amplitude).float(),
        indices=training_indices,
        enc_seq_len=args.enc_seq_len,
        dec_seq_len=args.dec_seq_len,
        target_seq_len=args.output_sequence_length,
        slice_size=training_slice_size
        )

    # Making dataloader
    training_time_data = DataLoader(training_time_data, args.batch_size, shuffle=False) #cannot shuffle time series

    model = tst.TimeSeriesTransformer(
        input_size=len(input_variables),
        batch_first=args.batch_first,
        num_predicted_features=len(input_variables) # 1 if univariate
        ).to(device)


    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = utils.generate_square_subsequent_mask(
        dim1=args.output_sequence_length,
        dim2=args.enc_seq_len
        ).to(device)

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.generate_square_subsequent_mask( 
        dim1=args.output_sequence_length,
        dim2=args.output_sequence_length
        ).to(device)

    # loss_fn = torch.nn.HuberLoss().to(device)
    loss_fn = torch.nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Define the warm-up schedule
    num_epochs = 50 # 50
    # total_steps = len(training_time_data) * num_epochs
    # Create the scheduler
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


    for epoch in range(num_epochs):
        output = train(model, training_time_data, src_mask, tgt_mask, loss_fn, optimizer, scheduler, args.batch_first, args.batch_size, input_size, args.LINEAR_DECODER)

        if epoch == num_epochs-1:
            print('hidden embeddings of epoch {}: {}'.format(epoch, output))
                
        if (epoch+1) % 10 == 0:
            # Save the model
            torch.save(model.state_dict(), 'model/model4D_{}_{}.pth'.format(args.enc_seq_len, args.output_sequence_length))
            # model.load_state_dict(torch.load('model.pth'))
