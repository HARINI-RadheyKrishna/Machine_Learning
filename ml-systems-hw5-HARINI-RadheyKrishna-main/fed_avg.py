import torch
import logging

from utils import (
    load_data,
    create_dataloader,
    set_random_seed,
    Net,
    data_split,
    plot_data_split,
)

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Aggregator:
    def __init__(self, model):
        self.model = model

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)

    def aggregate(self, state_dict_list, num_samples_list):
        # TODO: aggregate local model params using federated averaging algorithm
        # the weight of each local model is num of samples on that client / total num of samples
        # return the aggregated model params in the form of a state_dict


        return avg_params


if __name__ == "__main__":
    # fix randomness
    set_random_seed(42)

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Rank {rank} is using device {device}")

    # define model and move to device
    model = Net()
    model.to(device)

    # define optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_rounds = 5
    num_epochs = 1
    batch_size = 32

    # load cifar10 dataset
    train_data, test_data = load_data()

    # for q4, split both train_data and test_data evenly (iid)
    # for q5, split train_data unevenly (non-iid) and test_data evenly
    # here in simulation each client can see the whole dataset but only uses part of it
    # however, in practice, each client will have its distinct dataset and cannot access others' dataset
    split_method = "iid"  # options: "non-iid" or "iid"
    train_data_split = data_split(
        dataset=train_data, num_clients=size - 1, split_method=split_method
    )

    # plot the distribution of each client's dataset
    plot_data_split(
        data_split=train_data_split,
        num_clients=size - 1,
        num_classes=10,
        file_name=f"figures/{split_method}.png",
    )

    test_data_split = torch.utils.data.random_split(
        test_data,
        [len(test_data) // (size - 1) for _ in range(size - 1)],
    )

    if rank == 0:
        # initialize aggregator in the parameter server
        aggregator = Aggregator(model)

    # start federated learning
    for round_idx in range(num_rounds):
        # rank 0 is the parameter server and rest of the ranks are clients
        if rank == 0:
            # TODO: get global model params from the aggregator


            # TODO: send global model params to other ranks


            local_model_params_list = []
            num_samples_list = []
            # TODO: receive local model params from other ranks and append to local_model_params_list
            # and receive number of samples from other ranks and append to num_samples_list


            # TODO: aggregate local model params


            # TODO: set global model params to the aggregator


            logging.info(
                f"--------- | Round: {round_idx} | aggregation finished | ---------"
            )
        else:
            # TODO: create data loader on local dataset


            # TODO: receive global model params from parameter server

            
            # TODO: set local model params


            # start local training
            model.train()
            running_loss = 0

            for epoch_idx in range(num_epochs):
                for batch_idx, (data_local, target_local) in enumerate(train_loader):
                    data_local = data_local.to(device)
                    target_local = target_local.to(device)

                    output = model(data_local)
                    loss = criterion(output, target_local)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += loss.item()

            # evaluate model on testset
            model.eval()
            correct = 0
            total = 0
            accuracy = 0

            with torch.no_grad():
                for data_local, target_local in test_loader:
                    data_local = data_local.to(device)
                    target_local = target_local.to(device)

                    output = model(data_local)
                    _, predicted = torch.max(output.data, 1)
                    total += target_local.size(0)
                    correct += (predicted == target_local).sum().item()

                accuracy = correct / total
                running_loss = running_loss / len(train_loader) / num_epochs
                logging.info(
                    f"Client: {rank} | Round: {round_idx} | Loss: {running_loss:.4f} | Accuracy: {accuracy:.4f}"
                )

            # TODO: send local model params to parameter server


            # TODO: send number of samples to parameter server
