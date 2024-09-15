---
jupyter:
  accelerator: GPU
  colab:
    gpuType: T4
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="1" id="sKZFEUd7jgFt"}
``` python
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
```
:::

::: {.cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="6HytwdNcnVjK" outputId="7d3cac5d-9aa4-4db3-8497-bfdd378ac94d"}
``` python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

::: {.output .stream .stdout}
    Using cuda device
:::
:::

::: {.cell .markdown id="fEWcVcjPSmO3"}
# Data
:::

::: {.cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="BfFgIiPHnYd0" outputId="b97db237-3b5a-4723-905c-002b7a4aa0b6"}
``` python
def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    valid_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomCrop(120, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


train_loader, valid_loader = get_train_valid_loader(data_dir = './data',
                                                   batch_size = 64,
                                              augment = True,random_seed = 123)

test_loader = get_test_loader(data_dir = './data',
                              batch_size = 72)
```

::: {.output .stream .stdout}
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
:::

::: {.output .stream .stderr}
    100%|██████████| 170498071/170498071 [00:06<00:00, 28336023.92it/s]
:::

::: {.output .stream .stdout}
    Extracting ./data/cifar-10-python.tar.gz to ./data
    Files already downloaded and verified
    Files already downloaded and verified
:::
:::

::: {.cell .markdown id="CAac3x-pShe9"}
# Neural Network
:::

::: {.cell .code execution_count="4" id="88Snbw2cooBW"}
``` python
class alex_net(nn.Module):
    def __init__(self, num_classes=10):
        super(alex_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```
:::

::: {.cell .markdown id="UqAeIcEHSNkw"}
# Referencing the notebook from class:

## Create a function to define a loss function, optimizer, train the network and test

### SGD was proven to be better after comparison with RMSprop
:::

::: {.cell .code execution_count="5" id="Qy10KGcKOxhy"}
``` python
def tuning_alexnet(valid_loader, test_loader, train_loader,learning_rate,
                   num_classes, num_epochs,weight_decay,momentum):

    model = alex_net(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct/total))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct/total))
    return (correct/total)
```
:::

::: {.cell .markdown id="gG26eJ0HTEYz"}
# PSO for finding best hyperparameters
:::

::: {.cell .code execution_count="21" id="ckWVUEK1PeMr"}
``` python
num_classes=10
num_epochs=30

def particle_swarm_optimization(num_dimensions, num_particles, max_iter,
                                i_min=-10,i_max=10,bounds=None,w=0.5,c1=0.25,c2=0.75):

    if bounds is None:
        particles = [({'position': [np.random.uniform(i_min, i_max) for _ in range(num_dimensions)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(num_dimensions)],
                    'pbest': float('inf'),
                    'pbest_position': None})
                    for _ in range(num_particles)]
    else:
        particles = [({'position': [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(num_dimensions)],
                    'pbest': float('inf'),
                    'pbest_position': None})
                    for _ in range(num_particles)]

    gbest_value = float('inf')
    gbest_position = None

    for _ in range(max_iter):
        for particle in particles:
            position = particle['position']
            velocity = particle['velocity']
            current_value = tuning_alexnet(valid_loader, test_loader, train_loader,
                                           learning_rate=position[0], num_classes=num_classes,
                                           num_epochs=num_epochs,
                                           weight_decay=position[1], momentum=position[2])
            print(position, current_value)

            if 1-current_value < particle['pbest']:
                particle['pbest'] = current_value
                particle['pbest_position'] = position.copy()

            if 1-current_value < gbest_value:
                gbest_value = current_value
                gbest_position = position.copy()

            for i in range(num_dimensions):
                r1, r2 = np.random.uniform(), np.random.uniform()
                velocity[i] = w * velocity[i] + c1*r1 * (particle['pbest_position'][i]
                                                         - position[i]) + c2*r2 * (gbest_position[i]
                                                                                   - position[i])
                position[i] += velocity[i]
                if bounds is not None:
                    position[i] = np.clip(position[i],bounds[i][0],bounds[i][1])

    return gbest_position, gbest_value
```
:::

::: {.cell .markdown id="aqiBdMlifdam"}
# Results

## Accuracy did not reach over 84%. Need more adjustments, run time were extremely long {#accuracy-did-not-reach-over-84-need-more-adjustments-run-time-were-extremely-long}

## 30 epochs, 30 particles {#30-epochs-30-particles}

## the pso returns the following best values for an accuracy of 83.47% {#the-pso-returns-the-following-best-values-for-an-accuracy-of-8347}

### learning rate: 0.02457619672185781 {#learning-rate-002457619672185781}

### weight decay: 0.0001943732997406710 {#weight-decay-00001943732997406710}

### momentum: 0.49852329844216964 {#momentum-049852329844216964}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2Te-BhcOQMpF" outputId="a193978c-e2df-49ab-f98a-08a36bdb3971"}
``` python
particle_swarm_optimization(num_dimensions=3, num_particles=30, max_iter=10,
                            i_min=-0.001,i_max=0.001,bounds=[(0.002,0.02),
                             (0.0001,0.0002),(0.4,0.5)],w=0.2,c1=0.25,c2=0.75)
```

::: {.output .stream .stdout}
    Epoch [1/30], Step [704/704], Loss: 1.0347
    Epoch [2/30], Step [704/704], Loss: 1.6370
    Epoch [3/30], Step [704/704], Loss: 0.4174
    Epoch [4/30], Step [704/704], Loss: 0.2879
    Epoch [5/30], Step [704/704], Loss: 0.8862
    Epoch [6/30], Step [704/704], Loss: 0.0297
    Epoch [7/30], Step [704/704], Loss: 0.5063
    Epoch [8/30], Step [704/704], Loss: 0.5067
    Epoch [9/30], Step [704/704], Loss: 0.6071
    Epoch [10/30], Step [704/704], Loss: 1.2056
    Epoch [11/30], Step [704/704], Loss: 0.1369
    Epoch [12/30], Step [704/704], Loss: 0.4582
    Epoch [13/30], Step [704/704], Loss: 0.2994
    Epoch [14/30], Step [704/704], Loss: 0.2530
    Epoch [15/30], Step [704/704], Loss: 0.8787
    Epoch [16/30], Step [704/704], Loss: 0.5401
    Epoch [17/30], Step [704/704], Loss: 0.3267
    Epoch [18/30], Step [704/704], Loss: 0.3231
    Epoch [19/30], Step [704/704], Loss: 0.3159
    Epoch [20/30], Step [704/704], Loss: 0.3286
    Epoch [21/30], Step [704/704], Loss: 0.1119
    Epoch [22/30], Step [704/704], Loss: 0.3695
    Epoch [23/30], Step [704/704], Loss: 0.9328
    Epoch [24/30], Step [704/704], Loss: 0.5012
    Epoch [25/30], Step [704/704], Loss: 0.2348
    Epoch [26/30], Step [704/704], Loss: 0.5809
    Epoch [27/30], Step [704/704], Loss: 0.2163
    Epoch [28/30], Step [704/704], Loss: 0.6160
    Epoch [29/30], Step [704/704], Loss: 0.0261
    Epoch [30/30], Step [704/704], Loss: 0.2066
    Accuracy of the network on the 5000 validation images: 83.12 %
    Accuracy of the network on the 10000 test images: 83.47 %
    [0.02457619672185781, 0.00019437329974067105, 0.49852329844216964] 0.8347
    Epoch [1/30], Step [704/704], Loss: 1.4074
    Epoch [2/30], Step [704/704], Loss: 0.5075
    Epoch [3/30], Step [704/704], Loss: 0.7667
:::
:::
