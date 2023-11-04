import torch
import logging

from utils import load_data, create_dataloader, set_random_seed, Net


if __name__ == "__main__":
    # fix randomness
    set_random_seed(42)

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

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

    for epoch_idx in range(num_epochs):
        model.train()
        running_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # compute loss
            output = model(data)
            loss = criterion(output, target)

            # compute gradients
            loss.backward()

            # update parameters
            optimizer.step()

            # clear gradients
            optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                logging.info(
                    f"Epoch: {epoch_idx} Batch: {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f}"
                )
                running_loss = 0

    # evaluate model on testset after training
    model.eval()
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