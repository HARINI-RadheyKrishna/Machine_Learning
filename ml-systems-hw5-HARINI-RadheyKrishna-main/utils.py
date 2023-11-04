import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return train_data, test_data


def create_dataloader(train_data, test_data, batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def data_split(dataset, num_clients, split_method, alpha=0.1):
    if split_method == "iid":
        data_split = torch.utils.data.random_split(
            dataset, [len(dataset) // num_clients] * num_clients
        )

    elif split_method == "non-iid":
        min_size = 0
        num_classes = len(dataset.classes)

        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(np.array(dataset.targets) == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < len(dataset) / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )

                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        data_split = [
            torch.utils.data.Subset(dataset, idx_batch[i]) for i in range(num_clients)
        ]

    return data_split


def plot_data_split(data_split, num_clients, num_classes, file_name):
    data_per_client = []
    for i in range(num_clients):
        data_per_class = [0] * num_classes
        for idx in data_split[i].indices:
            label = data_split[i].dataset.targets[idx]
            data_per_class[label] += 1
        data_per_client.append(data_per_class)

    x = [f"client {i}" for i in range(1, num_clients + 1)]

    y = np.array(data_per_client)
    y_t = y.transpose()

    cmap = matplotlib.colormaps["tab10"]
    colors = cmap(np.arange(10))

    for i in range(10):
        plt.bar(
            x,
            y_t[i],
            bottom=np.sum(y_t[:i], axis=0),
            color=colors[i],
            label=f"Class {i}",
        )

    # put legent outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.subplots_adjust(right=0.8)

    plt.ylabel("Number of samples")
    plt.title("Data distribution")
    plt.savefig(f"{file_name}")
