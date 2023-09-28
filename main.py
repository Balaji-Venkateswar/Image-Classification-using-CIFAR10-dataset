
# %% imports

import os
import yaml
import torch
import torchvision

import matplotlib.pyplot as plt
import json

from statistics import mean
from model.conv import Net
from utils.create_folder import create_trainfolder
from utils.accuracy import accuracy

# %% config

yaml_file = open("./config/setup.yaml", 'r')
setup = yaml.load(yaml_file, Loader=yaml.FullLoader)
model_config = setup['model_config']

# %% create a function to create train folders

datas, config, img, model = create_trainfolder(os.getcwd())

# %% dataloader

dataset_path = os.path.join(os.path.dirname(os.getcwd()), '99_datasets', 'cifar10')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((setup['mean'], setup['mean'], setup['mean']), (setup['std'], setup['std'], setup['std']))])

trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                        download=setup['download'], transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=setup['batch_size'],
                                          shuffle=setup['shuffle'], num_workers=setup['num_workers'])
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                       download=setup['download'], transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=setup['batch_size'],
                                         shuffle=setup['shuffle'], num_workers=setup['num_workers'])

# %% load model

net = Net()

criterion = getattr(torch.nn, model_config['criterion'])()
optimizer = getattr(torch.optim, model_config['optimizer'])(net.parameters(), **model_config['optimizer_params'])

# %% training

losses, acc, mean_acc, mean_loss, test_loss, test_acc = [], [], [], [], [], []
batches = 0

for epoch in range(setup['epochs']):  # loop over the dataset multiple times
    print('EPOCH NUMBER IS ', epoch)
    for i, data in enumerate(trainloader, 0):
        correct = 0
        total = 0
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        print(labels.shape, outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_acc = 100 * correct // total
        batches += 1

        losses.append(loss.item())
        acc.append(train_acc)

        if batches % setup['batch_test'] == 0:
            mean_loss.append(mean(losses))
            mean_acc.append(mean(acc))
            test_los, test_accur = accuracy(net, testloader)
            test_loss.append(test_los.item())
            test_acc.append(test_accur)

print('Finished Training')

# %% store datalists

data_values = [losses, acc, mean_loss, mean_acc, test_loss, test_acc]
data_name = ['losses', 'acc', 'mean_train_loss',
             'mean_train_acc', 'test_loss', 'test_acc']
for counter, filename in enumerate(data_name, 0):
    with open(r'{}/{}.json'.format(datas, filename), 'w') as f:
        json.dump(data_values[counter], f)


# %% store config
with open(r'{}/config.yaml'.format(config), 'w') as file:
    yaml.dump(setup, file)

# %% store model
PATH = '{}/trained_model.pth'.format(model)
torch.save(net.state_dict(), PATH)

# %% store the plots

for count, i in enumerate(data_values, 0):
    fig = plt.figure()
    plt.plot(i)
    fig.savefig(os.path.join('{}'.format(img),
                '{}.png'.format(data_name[count])))
