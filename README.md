# How to code a Transformer autoencoder model for time series forecasting in PyTorch
## PyTorch implementation of Transformer model for time series forecasting on heterogeneous nodes with multiple features"

This is the repo of the Transformer autoencoder model for time series forecasting on KG

The sandbox.py file shows how to use the Transformer to make a training prediction on the data from the .csv file in "/data".

The inference_sandbox.py file contains the function that takes care of inference, and the inference_example.py file shows a pseudo-ish code example of how to use the function during model validation and testing. 

care about the learning rate to be smaller than 1e-5 if use the transformer decoder, otherwise may lead to overfitting.

## train
python sandbox.py

## test
python inference_sandbox.py

## pseudocode

    # Define the input dimensions
    batch_size = 32
    seq_len = 50
    embedding_dim = 128
    feature_dim = 3

    # Define the transformer model
    class TransformerModel(nn.Module):
        def __init__(self):
            super(TransformerModel, self).__init__()
        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )

        self.positional_decoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(dim_val)
            )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(dim_val)
            )

        def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None, linear_decoder: bool=False):
            # x.shape = (batch_size, seq_len, embedding_dim, feature_dim)
            # x = x.permute(1, 0, 3, 2) # reshape the input to (seq_len, batch_size, node_size, feature_dim)
            # x = x.reshape(seq_len, batch_size * node_size, feature_dim) # reshape the input to (seq_len, batch_size * node_size, feature_dim)
            return decoder_output
