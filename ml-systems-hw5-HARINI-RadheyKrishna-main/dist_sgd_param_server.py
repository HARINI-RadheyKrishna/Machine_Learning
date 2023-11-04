import torch
import logging

from utils import load_data, create_dataloader, set_random_seed, Net

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

    num_epochs = 3
    batch_size = 100

    # load cifar10 dataset
    train_data, test_data = load_data()
    train_loader, test_loader = create_dataloader(
        train_data, test_data, batch_size=batch_size
    )

    # compute number of samples for each rank
    # note that rank 0 is the parameter server, and we have size-1 workers
    num_samples_per_rank = int(batch_size / (size - 1))

    for epoch_idx in range(num_epochs):
        running_loss = 0

        for batch_idx, (data_global, target_global) in enumerate(train_loader):
            if rank == 0:
                data_split = torch.split(data_global, num_samples_per_rank, dim=0)
                target_split = torch.split(target_global, num_samples_per_rank, dim=0)

                # TODO: send local data and target to other ranks


                # TODO: receive loss from other ranks and sum them up


                # TODO: compute average loss by dividing number of global samples, and compute running loss


                # TODO: receive gradients from other ranks and sum them up


                # TODO: update parameters and send updated model to other ranks


                # clear gradients
                optimizer.zero_grad()

                # print statistics
                if (batch_idx + 1) % 200 == 0:
                    logging.info(
                        f"Epoch: {epoch_idx} Batch: {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f}"
                    )
                    running_loss = 0

            else:
                model.train()

                # TODO: receive data from rank 0


                # TODO: move data to device


                # TODO: compute loss


                # TODO: compute gradients


                # TODO: send loss to rank 0, this is not necessary but just for debugging
                # remember to multiply loss by number of local samples


                # TODO: send gradients to rank 0, remember to multiply gradients by number of local samples


                # TODO: receive updated model from rank 0


                # clear gradients
                optimizer.zero_grad()

    # evaluate model on testset after training on parameter server
    model.eval()
    if rank == 0:
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = correct / total
            logging.info(f"Test Accuracy: {accuracy:.4f}")