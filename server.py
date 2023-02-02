import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pickle

serverip = 'localhost'
serverport = 9999
buffer_size = 4096

# Downloading Dataset
def get_dataset():
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
    
    return training_set, test_set

# Creating dataloaders
def create_dataloader(training_set, test_set):
    batch_size = 4
    val_ratio = 0.2
    train_dataset, val_dataset = random_split(training_set, 
                                            [int((1 - val_ratio) * len(training_set)), 
                                            int(val_ratio * len(training_set))])
    train_dl = DataLoader(train_dataset, 
                        batch_size, 
                        shuffle=True, 
                        num_workers=3, 
                        pin_memory=True)
    valid_dl = DataLoader(val_dataset,
                        batch_size,
                        shuffle=True,
                        num_workers=3,
                        pin_memory=True)
    test_dl = DataLoader(test_set, 
                        batch_size, 
                        num_workers=3, 
                        shuffle=True, 
                        pin_memory=3)
    return train_dl, valid_dl, test_dl

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def create_socket_and_listen(serverIp, serverPort):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket successfully created")
        server.bind((serverIp, serverPort))
        server.listen(5)
        return server
    except socket.error as err:
        print(f"Socket creation failed with error {err}")

def send_tensor(filename, socket):
    buffer = 4096
    with open(filename, 'rb') as file:
        for buffer in file:
        #print(buffer)
            socket.send(buffer)
    socket.send(b"Done")
    return

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        return x
    
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ClientClassifier(nn.Module):
    def __init__(self):
        super(ClientClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x
    

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model1, model2):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs1 = model1(inputs)
        outputs2 = model2(outputs1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs2, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss
"""
# Socket Creation and Connection
server = create_socket_and_listen(serverip, serverport)
clientsocket, clientaddress = server.accept()
print(f'Connection with {clientaddress} established successfully')
"""
def main():
    # Model Initialization
    #clientmodel = ClientModel()
    #servermodel = ServerModel()
    #print('Weights: ', clientmodel.state_dict())
    model1 = ClientModel()
    model2 = ServerModel()

    # Sending the initial weights to the client
    #pickle.dump(clientmodel.state_dict(), open('server_weights.pkl', 'wb'))
    #send_tensor('server_weights.pkl', clientsocket)
    #print('Weights sent')

    # Device agnostic code
    device = get_default_device()
    device

    dataset, test_dataset = get_dataset()
    train_dl, valid_dl, test_dl = create_dataloader(dataset, test_dataset)

    loss_fn = torch.nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)

    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model1.train(True)
        model2.train(True)
        avg_loss = train_one_epoch(epoch_number, train_dl   , optimizer, loss_fn, model1, model2)

        # We don't need gradients on to do reporting
        model1.train(False)
        model2.train(False)
        
        running_vloss = 0.0
        for i, vdata in enumerate(valid_dl):
            vinputs, vlabels = vdata
            voutputs1 = model1(vinputs)
            voutputs2 = model2(voutputs1)
            vloss = loss_fn(voutputs2, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)

        epoch_number += 1

    
if __name__ == '__main__':
    main()