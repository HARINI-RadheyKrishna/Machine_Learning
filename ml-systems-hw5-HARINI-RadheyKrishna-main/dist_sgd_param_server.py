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
    train_loader, test_loader = create_dataloader(train_data, test_data, batch_size=batch_size)

    # compute number of samples for each rank
    num_samples_per_rank = int(batch_size / (size - 1))

    for epoch_idx in range(num_epochs):
        running_loss = 0


        for batch_idx, (data_global, target_global) in enumerate(train_loader):
            if rank == 0:
                data_split = torch.split(data_global, num_samples_per_rank, dim=0)
                target_split = torch.split(target_global, num_samples_per_rank, dim=0)

                # TODO: send local data and target to other ranks
                
                for i in range(1,size):
                    local_data = data_split[i-1]
                    local_target =target_split[i-1]
                    comm.send(local_data, dest = i, tag = 11)
                    comm.send(local_target, dest = i, tag = 22)

                loss = 0
                # TODO: receive loss from other ranks and sum them up
                
                for i in range(1,size):
                    loss += comm.recv(source = i, tag = 33)

                # TODO: compute average loss by dividing number of global samples, and compute running loss
                
                loss /= data_global.shape[0]
                running_loss += loss.item()

                # TODO: receive gradients from other ranks and sum them up
                
                
                for param in model.parameters():
                    param.grad = torch.zeros_like(param.data)
                    for i in range(1, size):
                        param.grad += comm.recv(source=i, tag=44)
                    param.grad /= data_global.shape[0]

                # TODO: update parameters and send updated model to other ranks
                
                
                optimizer.step()
                for i in range(1,size):
                    comm.send(model, dest = i, tag = 55)

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

                # Receive data from rank 0
                local_data = comm.recv(source=0, tag=11)
                local_target = comm.recv(source=0, tag=22)

                # Move data to device
                local_data = local_data.to(device)
                local_target = local_target.to(device)

                # Compute loss
                output = model(local_data)
                total_loss = criterion(output, local_target)

                # Compute gradients
                total_loss.backward()

                # Send loss to rank 0, this is not necessary but just for debugging
                # Remember to multiply loss by the number of local samples
                total_loss *= local_data.shape[0]
                comm.send(total_loss, dest = 0, tag = 33)
                # Send gradients to rank 0, remember to multiply gradients by the number of local samples
                for param in model.parameters():
                    param.grad *= local_data.shape[0]
                    comm.send(param.grad, dest = 0, tag = 44)

                # Receive updated model from rank 0
                model = comm.recv(source = 0, tag = 55)

                # Clear gradients
                optimizer.zero_grad()

    # Evaluate model on the test set after training on the parameter server
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
