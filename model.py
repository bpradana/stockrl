import torch.nn as nn


class LSTM_QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_QNetwork, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Hidden layer
        self.hidden = nn.Linear(hidden_size, hidden_size)

        # Fully connected layer to produce Q-values
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # We take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Pass the output through the hidden layer
        hidden_out = self.hidden(lstm_out)

        # Pass the output through the fully connected layer to get Q-values
        q_values = self.fc(hidden_out)

        return q_values
