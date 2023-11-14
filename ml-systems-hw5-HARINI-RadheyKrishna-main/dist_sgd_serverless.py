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
    num_samples_per_rank = int(batch_size / size)

    for epoch_idx in range(num_epochs):
        model.train()
        running_loss = 0

        for batch_idx, (data_global, target_global) in enumerate(train_loader):
            data_split = torch.split(data_global, num_samples_per_rank, dim=0)
            target_split = torch.split(target_global, num_samples_per_rank, dim=0)

            data_local = data_split[rank].to(device).requires_grad_()
            target_local = target_split[rank].to(device)

            # compute loss
            output = model(data_local)
            loss = criterion(output, target_local)

            # compute gradients
            loss.backward()

            # TODO: reduce loss to rank 0, this is not necessary but just for debugging
            # remember to multiply loss by number of local samples before reduce
            multiplied_loss = num_samples_per_rank * loss
            loss = comm.reduce(multiplied_loss, root=0)

            # TODO: allreduce gradients for each layer
            # remember to multiply gradients by number of local samples before allreduce
            # and divide gradients by number of global samples after allreduce
            for param in model.parameters():
                param.grad *= data_local.shape[0]
                grad = param.grad.clone()  
                reduced_gradients = comm.allreduce(grad, op=MPI.SUM)
                reduced_gradients /= data_global.shape[0]
                param.grad = reduced_gradients

            # update parameters
            optimizer.step()

            # clear gradients
            optimizer.zero_grad()

            # print statistics
            if rank == 0:
                loss /= data_global.shape[0]
                #loss = comm.reduce(loss, op=MPI.SUM, root=0)
                running_loss += loss.item()

                if (batch_idx + 1) % 200 == 0:
                    logging.info(
                        f"Epoch: {epoch_idx} Batch: {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f}"
                    )
                    running_loss = 0

    # evaluate model on testset after training on rank 0
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