# How to code a Transformer autoencoder model for time series forecasting in PyTorch
## PyTorch implementation of Transformer model, refered to the implementation of the paper: "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case"

This is the repo of the Transformer autoencoder model for time series forecasting

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
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
            self.fc = nn.Linear(embedding_dim * seq_len, 1)

        def forward(self, x):
            # x.shape = (batch_size, seq_len, embedding_dim, feature_dim)
            x = x.permute(1, 0, 3, 2) # reshape the input to (seq_len, batch_size, feature_dim, embedding_dim)
            x = x.reshape(seq_len, batch_size * feature_dim, embedding_dim) # reshape the input to (seq_len, batch_size * feature_dim, embedding_dim)
            x = self.transformer_encoder(x)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x
