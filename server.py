import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

import utils

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()        
        self.conv1 = nn.Conv1d(1, 16, 7)  # 124 x 16        
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 62 x 16
        self.conv2 = nn.Conv1d(16, 16, 5)  # 58 x 16
        self.relu2 = nn.LeakyReLU()        
        self.conv3 = nn.Conv1d(16, 16, 5)  # 54 x 16
        self.relu3 = nn.LeakyReLU()        
        self.conv4 = nn.Conv1d(16, 16, 5)  # 50 x 16
        self.relu4 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool1d(2)  # 25 x 16
        self.linear5 = nn.Linear(25 * 16, 128)
        self.relu5 = nn.LeakyReLU()        
        self.linear6 = nn.Linear(128, 5)
        self.softmax6 = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)        
        x = self.conv2(x)
        x = self.relu2(x)        
        x = self.conv3(x)
        x = self.relu3(x)        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(-1, 25 * 16)
        x = self.linear5(x)
        x = self.relu5(x)        
        x = self.linear6(x)
        x = self.softmax6(x)
        return x

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--log_host",
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="Net",
    choices=["Net"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="training batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    help="number of epochs of training",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()


def main() -> None:
    """Start server and train five rounds."""

    print(args)

    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    _, testset = utils.load_ecg(cid=0)

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
        evaluate_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    # Run server
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(server_round),
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=False)


def get_eval_fn(testset) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluatee(self, weights: fl.common.NDArrays, Optional) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluatee


if __name__ == "__main__":
    main()