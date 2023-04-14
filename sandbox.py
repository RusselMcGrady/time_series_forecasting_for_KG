"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import util.dataset as ds
import util.utils as utils
import torch
import datetime
import time
import layer.TransformerGAT as tst
import numpy as np
import math
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import MinMaxScaler, StandardScaler


torch.manual_seed(42)
np.random.seed(42)

SCALER = True
LINEAR_DECODER = False

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, training_time_data, src_mask, tgt_mask, loss_fn, optimizer, scheduler, batch_first, batch_size, input_size):
    model.train() # Turn on the train mode \o/
    start_time = time.time()
    total_loss = 0.

    for step, batch in enumerate(training_time_data):
        optimizer.zero_grad()
        
        src, trg, trg_y = batch
        B, N, T, F = trg_y.size()
        if input_size == 1:
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
            
                loss = loss_fn(output, trg_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        ave_node_loss = total_loss / N
        elapsed = time.time() - start_time
        print('| epoch {:3d} | step {:3d} | '
                'lr {:02.8f} | {:5.2f} ms | '
                'batch node loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, step, scheduler.get_last_lr()[0], # get_lr()
                elapsed * 1000, ave_node_loss, math.exp(ave_node_loss))) # math.log(cur_loss)
        
    total_loss = 0
    start_time = time.time()
    scheduler.step()

    # print('-'*12)
    # print(scaler.inverse_transform(output.reshape(-1, 5).detach().cpu()))
    # print('-'*12)
    # print(scaler.inverse_transform(trg_y.reshape(-1, 5).detach().cpu()))
    # print('-'*12)
    return output

# Hyperparams
test_size = 0.2
batch_size = 32
target_col_name = "Reliability"
timestamp_col = "Timestamp"
node_col = "Node"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 2 # length of input given to decoder 92
enc_seq_len = 5 # length of input given to encoder 153
output_sequence_length = 2 # target sequence length. If hourly data and length = 48, you predict 2 days ahead 48
if LINEAR_DECODER:
    output_sequence_length = enc_seq_len
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = ['Flexibility','Service','Infrastructure Quality','Freight'] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
# Input x
# (batch_size, nodes, sequentials, features)
data, slice_size = utils.read_data(file_name='forecast_cyclical_data', node_col_name=node_col, timestamp_col_name=timestamp_col)

# Remove test data from dataset for each node
ratio = round(slice_size*(1-test_size))
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
    step_size=step_size,
    slice_size=training_slice_size)

# looks like normalizing input values curtial for the model
scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = StandardScaler()
# Recover the original values
# original_data = scaler.inverse_transform(scaled_data)
series = training_time_data[input_variables].values

# for i, val in enumerate(series.mean(axis=0)):
#      series[:,i][series[:,i] == 0] = val
# series = np.where(series == 0, 1e-9, series)

if SCALER:
    amplitude = scaler.fit_transform(series)
else:
    amplitude = series
    
# Making instance of custom dataset class
training_time_data = ds.TransformerDataset( ## todo
    data=torch.tensor(amplitude).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length,
    slice_size=training_slice_size
    )

# Making dataloader
training_time_data = DataLoader(training_time_data, batch_size, shuffle=False) #cannot shuffle time series

model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    batch_first=batch_first,
    num_predicted_features=len(input_variables) # 1 if univariate
    ).to(device)


# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    ).to(device)

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=output_sequence_length,
    dim2=output_sequence_length
    ).to(device)

# loss_fn = torch.nn.HuberLoss().to(device)
loss_fn = torch.nn.MSELoss().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Define the warm-up schedule
num_epochs = 1000 # 50
# total_steps = len(training_time_data) * num_epochs
# Create the scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


for epoch in range(num_epochs):
    output = train(model, training_time_data, src_mask, tgt_mask, loss_fn, optimizer, scheduler, batch_first, batch_size, input_size)

    if epoch == num_epochs-1:
        print('hidden embeddings of epoch {}: {}'.format(epoch, output))
            
    if (epoch+1) % 10 == 0:
        # Save the model
        torch.save(model.state_dict(), 'model/model4D_{}_{}.pth'.format(enc_seq_len, output_sequence_length))
        # model.load_state_dict(torch.load('model.pth'))