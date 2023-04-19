"""
code-ish example of how to use the inference function to do validation
during training. 

The validation loop can be used as-is for model testing as well.

NB! You cannot use this script as is. This is merely an example to show the overall idea - 
not something you can copy paste and expect to work. For instance, see "sandbox.py" 
for example of how to instantiate model and generate dataloaders.

If you have never before trained a PyTorch neural network, I suggest you look
at some of PyTorch's beginner-level tutorials.
"""
import torch
import argparse
import util.inference as inference
import util.utils as utils
import layer.TransformerGAT as tst
import numpy as np
import util.dataset as ds
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(model, test_time_data, src_mask, tgt_mask, loss_fn, batch_first, batch_size, input_size, forecast_window, PLOT_BIAS, PLOT_PREDICT, SCALER, LINEAR_DECODER):
    # Set the model to evaluation mode
    model.eval()
    output = torch.Tensor(0)    
    truth = torch.Tensor(0)
    output_scale = torch.Tensor(0)
    truth_scale = torch.Tensor(0)
    total_loss = 0.

    for step, batch in enumerate(test_time_data):

        src, trg, trg_y = batch
        B, N, T, F = trg_y.size()
        if input_size == 1: # feature size = 1
            trg_y.unsqueeze(2)
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
        
        # Split the data tensor along dimension 1 into smaller tensors
        src_chunks, trg_chunks, trg_y_chunks = src.chunk(src.size(1) // batch_size, dim=1), trg.chunk(trg.size(1) // batch_size, dim=1), trg_y.chunk(trg_y.size(1) // batch_size, dim=1)
        # Create a list of TensorDatasets
        minibatch_datasets = [TensorDataset(src_chunk, trg_chunk, trg_y_chunk) for src_chunk, trg_chunk, trg_y_chunk in zip(src_chunks, trg_chunks, trg_y_chunks)]

        # Create a list of DataLoaders, shuffle node mini batches
        minibatch_dataloaders = [DataLoader(minibatch_dataset, batch_size=batch_size, shuffle=False) for minibatch_dataset in minibatch_datasets]

        for minibatch_dataloader in minibatch_dataloaders:
            for _, (src, trg, trg_y) in enumerate(minibatch_dataloader):

                # Permute from shape [batch size, node size, seq len, num features] to [seq len, batch size*node size, num features]
                # Node dimension is put inside the batch, in order to process each node along the time separately
                if batch_first == False:
                    src = src.permute(2, 0, 1, 3)
                    src = src.reshape(src.size()[0], src.size()[1] * src.size()[2], src.size()[3])
                    # print("src shape changed from {} to {}".format(shape_before, src.shape))

                    trg = trg.permute(2, 0, 1, 3)
                    trg = trg.reshape(trg.size()[0], trg.size()[1] * trg.size()[2], trg.size()[3])

                    trg_y = trg_y.permute(2, 0, 1, 3)
                    trg_y = trg_y.reshape(trg_y.size()[0], trg_y.size()[1] * trg_y.size()[2], trg_y.size()[3])

                # inference on the length of the output window
                prediction = model(
                    src=src,
                    tgt=trg,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    linear_decoder=LINEAR_DECODER
                )

                if batch_first == False: # must permute the dimension for plot
                    prediction = prediction.permute(1, 0, 2)
                    trg_y = trg_y.permute(1, 0, 2)

                total_loss += loss_fn(prediction, trg_y).item()

                # for evaluating metrics
                output_scale = torch.cat((output_scale, prediction.reshape(-1, input_size).detach().cpu()))
                truth_scale = torch.cat((truth_scale, trg_y.reshape(-1, input_size).detach().cpu()))

                # operation reshape(-1, input_size) concat the sequential data in  the form of time series
                if SCALER:
                    output = torch.cat((output, torch.tensor(scaler.inverse_transform(prediction.reshape(-1, input_size).detach().cpu())).cpu()), 0)
                    truth = torch.cat((truth, torch.tensor(scaler.inverse_transform(trg_y.reshape(-1, input_size).detach().cpu())).contiguous().cpu()), 0)
                else:
                    output = torch.cat((output, prediction.reshape(-1, input_size).cpu()), 0)
                    truth = torch.cat((truth, trg_y.reshape(-1, input_size).contiguous().cpu()), 0)    
               

    if PLOT_BIAS == True:
        plot_length = len(output)
        # plot_length = 100
        utils.plot(output, truth, step, plot_length, total_loss/plot_length)
    
    if PLOT_PREDICT == True:
        if batch_first == False:
            batch_size = src.size()[1]
        else:
            batch_size = src.size()[0]
        # inference one by one
        forecast = inference.run_encoder_decoder_inference(
            model=model, 
            src=src, 
            forecast_window=forecast_window,
            batch_size=batch_size,
            device=device,
            batch_first=batch_first
            ) # predict forecast_window steps
        
        if batch_first == False: # must permute the dimension for plot
            forecast = forecast.permute(1, 0, 2)
            src = src.permute(1, 0, 2)
        if SCALER: # recover scaled data
            src = torch.tensor(scaler.inverse_transform(src[-1,:,].detach().cpu()))
            forecast = torch.tensor(scaler.inverse_transform(forecast[-1,:,].detach().cpu()))
        else:
            src = src[-1,:,].cpu()
            forecast = forecast[-1,:,].cpu()

        utils.predict_future(src.unsqueeze(0), forecast.unsqueeze(0))

    # reshape(B, N, T, F)
    # print('output tensor in shape (N, B, T, F): \n',output.reshape(B, N, T, F))
    utils.evaluate_forecast(truth_scale, output_scale)
        

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
    argparser.add_argument("--PLOT_BIAS", type=bool, default=True)
    argparser.add_argument("--PLOT_PREDICT", type=bool, default=True)
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
    argparser.add_argument("--forecast_window", type=int, default=4,
                           help="window you forecast in future")
    argparser.add_argument("--step_size", type=int, default=1,
                           help="Step size, i.e. how many time steps does the moving window move at each step")
    argparser.add_argument("--in_features_encoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--in_features_decoder_linear_layer", type=int, default=2048)
    argparser.add_argument("--batch_first", type=bool, default=False)
    argparser.add_argument("--target_col_name", type=str, default="Reliability")
    argparser.add_argument("--timestamp_col", type=str, default="Timestamp")
    argparser.add_argument("--node_col", type=str, default="Node")
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

    # Get test data from dataset
    ratio = round(slice_size*args.test_size)
    first_round = data.iloc[slice_size-ratio:slice_size, :]
    for i in range(1,round(len(data)//slice_size)+1):
        first_round = pd.concat([first_round, data.iloc[slice_size*(i+1)-ratio:slice_size*(i+1), :]], axis=0)
    test_time_data = first_round
    test_slice_size = ratio
    # test_time_data = data[-(round(len(data)*test_size)):]

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
    # Should be test data indices only
    test_indices = utils.get_indices_input_target(
        num_obs=len(test_time_data), # round(len(data)*test_size)
        input_len=window_size,
        step_size=window_size,
        forecast_horizon=0,
        target_len=args.output_sequence_length,
        slice_size=test_slice_size
    )

    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    # Recover the original values
    # original_data = scaler.inverse_transform(scaled_data)
    map_series = test_time_data[input_variables].values
    labels = test_time_data["Node Label"].values
    
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
    test_time_data = ds.TransformerDataset(
        data=torch.tensor(amplitude).float(),
        indices=test_indices,
        enc_seq_len=args.enc_seq_len,
        dec_seq_len=args.output_sequence_length,
        target_seq_len=args.output_sequence_length,
        slice_size=test_slice_size
        )

    # Making dataloader
    test_time_data = DataLoader(test_time_data, args.batch_size)

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

    # Initialize the model with the same architecture and initialization as when it was saved
    model = tst.TimeSeriesTransformer(
        input_size=len(input_variables),
        dec_seq_len=args.enc_seq_len,
        batch_first=args.batch_first,
        num_predicted_features=len(input_variables) # 1 if univariate
        ).to(device)

    # Define the file path, same as the forecast_window
    PATH = 'model/model4D_{}_{}.pth'.format(args.enc_seq_len, args.output_sequence_length)

    # Load the saved state dictionary into the model
    model.load_state_dict(torch.load(PATH))
    # Load the state dict into the model
    # state_dict  = torch.load(PATH, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    # loss_fn = torch.nn.HuberLoss().to(device)
    loss_fn = torch.nn.MSELoss().to(device)



    # Iterate over all (x,y) pairs in validation dataloader
    with torch.no_grad():
        eval(model, test_time_data, src_mask, tgt_mask, loss_fn, args.batch_first, args.batch_size, input_size, args.forecast_window, args.PLOT_BIAS, args.PLOT_PREDICT, args.SCALER, args.LINEAR_DECODER)

