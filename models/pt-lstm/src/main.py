import numpy as np

import pandas as pd
import torch

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class LongShortTermMemory(torch.nn.Module):
    """Parameters
    ----------
    season_length : int
        Number of observations per unit of time. Ex: 24 Hourly data.
    decomposition_type : str
        Sesonal decomposition type, 'multiplicative' (default) or 'additive'.
    model : str
        Controlling Theta Model. By default searchs the best model. ["STM", "OTM", "DSTM", "DOTM"]
    alias : str
        Custom name of the model.
    prediction_intervals : Optional[ConformalIntervals]
        Information to compute conformal prediction intervals.
        By default, the model will compute the native prediction
        intervals.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 batch_size=64,
                 bias=True,
                 batch_first=False,
                 dropout=0.0,
                 lr=0.001,
                 train_step=100,
                 alias="lstm",
                 prediction_intervals=None
                 ):
        super(LongShortTermMemory, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.batch_size = batch_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.alias = alias
        self.prediction_intervals = prediction_intervals

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Training loop
        self.train_step = train_step

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="torch.nn.LSTM",
            authors=[AuthorInfo(name="Antonio Gallo", email="antonio.gallo@polito.it"),
                     AuthorInfo(name="Giacomo Buscemi", email="giacomo.buscemi@polito.it")],
            type=ForecasterType.quantile,
            params={
                self.input_size,
                self.hidden_size,
                self.num_layer,
                self.bias,
                self.batch_first,
                self.dropout,
                self.alias
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates = None,
        future_covariates = None,
        level = None,
        ** kwargs,
    ) -> pd.DataFrame:

        y = history.y.fillna(history.y.mean())

        # Make forecast
        """Parameters
        y : numpy.array
            Clean time series of shape (n, ) to train the model.
        h : int
            Lenght of prediction horizon.
        X : array-like
            Optional insample exogenous of shape (t, n_x).
        X_future : array-like
            Optional exogenous of shape (h, n_x)
        fitted : bool
            Whether or not returns insample predictions"""
        losses = []

        norm_y = (y - y.min()) / (y.max() - y.min())

        # TODO: Here past covariates should be included into the observed history and normalized

        for step in range(self.train_step):

            idx = np.random.randint(0, len(y) - self.input_size - horizon, self.batch_size)

            # Initialize a torch tensor with the batch size and input size
            x = torch.zeros(self.batch_size, self.input_size)
            labels = torch.zeros(self.batch_size, horizon)

            for i, id in enumerate(idx):
                # Forward pass
                x[i, :] = torch.tensor(norm_y[id:id + self.input_size].values)
                labels[i, :] = torch.tensor(norm_y[id + self.input_size:id + self.input_size+horizon].values)

            outputs = self.forward(torch.tensor(x))
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for param in self.lstm.parameters():
            param.data = param.data.to(torch.float32)

        pred = self.forward(torch.tensor(norm_y.values[-self.input_size:]).unsqueeze(0).to(torch.float32)).detach().numpy().squeeze()  # This function should output the predictions for the next h steps given y as training set

        # Denormalize predictions
        pred = pred * (y.max() - y.min()) + y.min()

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Format forecast dataframe
        prediction = (
            pd.DataFrame(
                index=index,
                data=pred
            )
            .rename_axis("ds")
            .rename(columns={0: "yhat"})
            .fillna(y.mean())
        )

        return prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


# Instantiate your model
model = LongShortTermMemory(48,
                            hidden_size=64,
                            output_size=2 * n_hours_horizon,
                            num_layers=2,
                            bias=True,
                            train_step=500,
                            batch_first=True,
                            dropout=0.2)

# Create a forecast server by passing in your model
app = server_factory(model)
