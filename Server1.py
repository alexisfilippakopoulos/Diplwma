import pyodbc
import socket
import pickle
import threading
import torch
from torch import nn as nn
import struct
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

# Events to enasure data is saved in the db before trying to access it
training_batches_event = threading.Event()
training_outputs_event = threading.Event()
training_labels_event = threading.Event()
training_loss_event = threading.Event()
validation_batches_event = threading.Event()
validation_outputs_event = threading.Event()
validation_labels_event = threading.Event()
updated_model_weights_event = threading.Event()
updated_class_weights_event = threading.Event()

event = threading.Event()
training_event = threading.Event()
training_lock = threading.Lock()
cursor_lock = threading.Lock()
select_event = threading.Event()

class Server:
    def __init__(self):
        ##super().__init__()
        self.ip = 'localhost'
        self.port = 9999
        self.fetching_connection = pyodbc.connect(
            'Driver={ODBC Driver 17 for SQL Server};'
            'Server=LAPTOP-LGNU4S88;'
            'Database=Thesis;'
            'Trusted_Connection=yes;'
        )
        self.storing_connection = pyodbc.connect(
            'Driver={ODBC Driver 17 for SQL Server};'
            'Server=LAPTOP-LGNU4S88;'
            'Database=Thesis;'
            'Trusted_Connection=yes;'
        )
        training_event.set()
        self.storing_cursor = self.storing_connection.cursor()
        self.fetching_cursor = self.fetching_connection.cursor()
        self.client_trained_counter = 0
        self.client_ids = []
        threading.Thread(target=self.create_socket, args=(())).start()
        

    def create_socket(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f'\n[+] Server socket created successfully at {self.ip, self.port}\n')
            server.bind((self.ip, self.port))
            server.listen()
            listen_thread = threading.Thread(target=self.listen_for_connections, args=(server,))
            listen_thread.start()
        except socket.error as error:
            print(f'Socket creation failed with error: {error}')
            server.close()


    def send_data(self, socket, data):
        serialized_tensor = pickle.dumps(data)
        socket.sendall(serialized_tensor)
        socket.sendall(b'<Done>')
        return
    
    def listen_for_connections(self, server):
        while True:
            client_socket, client_address = server.accept()
            print(f'[+] Connection with {client_address} established.\n[+] Connected Clients: {len(self.client_ids)}')
            with cursor_lock:
                self.storing_cursor.execute('INSERT INTO clients(address, port) VALUES (?, ?)', (client_address[0], client_address[1]))
                self.storing_connection.commit()
                self.storing_cursor.execute('SELECT id FROM clients WHERE address = ? AND port = ?', client_address[0], client_address[1])
                client_id = self.storing_cursor.fetchone()[0]
            self.client_ids.append(client_id)
            communication_thread = threading.Thread(target=self.listen_for_data, args=(client_socket, client_address, client_id))
            communication_thread.start()
            client_thread = threading.Thread(target=self.client_handler, args=(client_socket, client_address, client_id))
            client_thread.start()
        
    def listen_for_data(self, client_socket, client_address, client_id):
        data = b''
        while True:
            data_chunk = client_socket.recv(4096)
            if str(data_chunk).__contains__('<SEPERATOR>') and str(data_chunk).__contains__('Done'):
                payload = data_chunk.split(b'<SEPERATOR>')[1].split(b'Done')[0]
                payload = bytes(payload)
                data += payload
                if header.__contains__('train'):
                    table = 'training'
                elif header.__contains__('valid'):
                    table = 'validation'
                elif header.__contains__('weights'):
                    teable = 'weights'
                if header.__contains__('batch'):
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, True, table, client_id))
                    storing_thread.start()
                else:
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, False, table, client_id))
                    storing_thread.start()
                
                client_socket.send(bytes('<Recvd>', 'utf-8'))
                data = b''
            elif str(data_chunk).__contains__('<SEPERATOR>'):
                header, payload = data_chunk.split(b'<SEPERATOR>', 1)
                header = header.decode()
                if payload != b'':
                    payload = bytes(payload)
                    data += payload
            elif str(data_chunk).__contains__('<Done>'):
                payload, _ = data_chunk.split(b'<Done>', 1)
                payload = bytes(payload)
                data += payload
                if header.__contains__('train'):
                    table = 'training'
                elif header.__contains__('valid'):
                    table = 'validation'
                elif header.__contains__('weights'):
                    teable = 'weights'
                if header.__contains__('batch'):
                    #table = 'training' if header.__contains__('train') else 'validation'
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, True, table, client_id))
                    storing_thread.start()
                elif header.__contains__('weights'):
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, False, 'weights', client_id))
                    storing_thread.start()
                else:
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, False, table, client_id))
                    storing_thread.start()
                client_socket.send(bytes('<Recvd>', 'utf-8'))
                data = b''
            else:
                data += data_chunk

            if not data_chunk:
                break

    def store_data(self, client_address, data, header, int_flag, table, client_id):
        query = f"""
                IF EXISTS (SELECT * FROM {table} WHERE client_id = ?)
                BEGIN
                    UPDATE {table} SET {header} = ? WHERE client_id = ?
                END
                ELSE
                BEGIN
                    INSERT INTO {table} (client_id, {header}) VALUES (?, ?)
                END"""
        if int_flag:
            with cursor_lock:
                self.storing_cursor.execute(query, (client_id, int(struct.unpack('!i', data)[0]), client_id, client_id, int(struct.unpack('!i', data)[0])))
        else:
            with cursor_lock:
                self.storing_cursor.execute(query, (client_id, data, client_id, client_id, data))
        
        self.storing_connection.commit()

        if header == 'training_batches': 
            training_batches_event.set()
        if header == 'training_outputs': 
            training_outputs_event.set()
        if header == 'training_labels': 
            training_labels_event.set()
        if header == 'training_loss': 
            training_loss_event.set()
        if header == 'validation_batches': 
            validation_batches_event.set()
        if header == 'validation_outputs': 
            validation_outputs_event.set()
        if header == 'validation_labels': 
            validation_labels_event.set()
        if header == 'model_updated_weights': 
            updated_model_weights_event.set()
        if header == 'classifier_updated_weights': 
            updated_class_weights_event.set()

    def aggregate_models(self, client_id, client_cursor, client_connection):
        client_cursor.execute("SELECT model_updated_weights FROM weights WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")") 
        models = client_cursor.fetchall()
        client_cursor.execute("SELECT training_batches FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.client_ids) + ")")                    
        datasizes = client_cursor.fetchall()
        # Get all the model weights into a list
        weighted_average = {}
        state_dicts = []
        for row in models:
            if row[0] is not None:
                state = pickle.loads(row[0])
                state_dicts.append(state)
        # Get total data size in order to find relevant one for each client
        total_data_size = 0
        data_per_client = []
        for row in datasizes:
            if row[0] is not None:
                data_per_client.append(row[0])
                total_data_size += row[0]

        # Compute weighted average of all the clients' weights
        for i in range(len(state_dicts)):
            for key in state_dicts[i].keys():
                if key not in weighted_average:
                    weighted_average[key] = state[key] * (data_per_client[i] / total_data_size)
                else:
                    weighted_average[key] += state[key] * (data_per_client[i] / total_data_size)
        
        # Retreive client's weights
        client_cursor.execute('SELECT model_updated_weights FROM weights WHERE client_id = ?', client_id)
        updated_weights = pickle.loads(client_cursor.fetchone()[0])

        # Compute aggregated client weights
        updated_weights = {key: value * 0.5 for key,value in updated_weights.items()}
        weighted_average = {key: value * 0.5 for key,value in weighted_average.items()}
        aggregated_weights = {key: updated_weights[key] + weighted_average[key] for key in updated_weights}
        client_cursor.execute('UPDATE weights SET model_aggregated_weights = ? WHERE client_id = ?', pickle.dumps(aggregated_weights), client_id)
        client_connection.commit()

        print(f'aggregated model for client_id: {client_id}')


    def client_handler(self, client_socket, client_address, client_id):
        # Model Initialization
        client_model = ClientModel()
        server_model = ServerModel()
        self.send_data(client_socket, client_model.state_dict())
        query = f"""
                IF EXISTS (SELECT * FROM weights WHERE client_id = ?)
                BEGIN
                    UPDATE weights SET model_aggregated_weights = ? WHERE client_id = ?
                END
                ELSE
                BEGIN
                    INSERT INTO weights (client_id, model_aggregated_weights) VALUES (?, ?)
                END"""
        with cursor_lock:
            self.storing_cursor.execute(query, client_id, pickle.dumps(client_model.state_dict()), client_id, client_id, pickle.dumps(client_model.state_dict()))
            self.storing_connection.commit()
        print('Weights sent\n')
        client_connection = pyodbc.connect(
            'Driver={ODBC Driver 17 for SQL Server};'
            'Server=LAPTOP-LGNU4S88;'
            'Database=Thesis;'
            'Trusted_Connection=yes;'
        )
        client_cursor = client_connection.cursor()
        # Device agnostic code
        device = self.get_default_device()
        device
        # Define loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
        # Start Training
        EPOCHS = 20
        self.train(EPOCHS, server_model, optimizer, loss_fn, client_socket, client_address, client_id, client_cursor, client_connection)


    def get_default_device(self, ):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    #train_one_epoch(epoch, optimizer, loss_fn, server_model, clientsocket, client_batches, client_address)
    def train_one_epoch(self, optimizer, loss_fn, servermodel, clientsocket, client_batches, client_address, client_id, epoch, client_cursor, client_connection):
        running_loss = 0.
        for i in range(client_batches):
            optimizer.zero_grad()
            training_outputs_event.wait()
            with cursor_lock:
                client_cursor.execute('SELECT training_outputs FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_outputs = pickle.loads(client_cursor.fetchone()[0])
                client_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_outputs_event.clear()
            outputs2 = servermodel(client_outputs)
            training_labels_event.wait()
            with cursor_lock:
                client_cursor.execute('SELECT training_labels FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_labels = pickle.loads(client_cursor.fetchone()[0])
                client_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_labels_event.clear()
            loss = loss_fn(outputs2, client_labels)
            training_loss_event.wait()
            with cursor_lock:
                client_cursor.execute('SELECT training_loss FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_loss = pickle.loads(client_cursor.fetchone()[0])
                client_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_loss_event.clear()
            global_loss = (0.5 * client_loss) + (0.5 * loss)
            self.send_data(clientsocket, global_loss)
            global_loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            # Gather data and report
            running_loss += loss.item()
        
        return running_loss / client_batches

    def train(self, epochs, server_model, optimizer, loss_fn, clientsocket, client_address, client_id, client_cursor, client_connection):

        best_vloss = 1_000_000.
        for epoch in range(0, epochs):
            training_batches_event.wait()
            with cursor_lock:
                client_cursor.execute(f'SELECT training_batches FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_batches = client_cursor.fetchone()[0]
                client_connection.commit()
                #self.fetching_cursor.close()
                #self.fetching_cursor = self.fetching_connection.cursor()
            training_batches_event.clear()
            server_model.train()
            with training_lock:
                clientsocket.send(b'<Train>')
                print(f'Training with {client_address} intiated\n')
                avg_loss = self.train_one_epoch(optimizer, loss_fn, server_model, clientsocket, client_batches, client_address, client_id, epoch, client_cursor, client_connection)
                # We don't need gradients on to do reporting
                updated_model_weights_event.wait()
                updated_model_weights_event.clear()
                updated_class_weights_event.wait()
                updated_class_weights_event.clear()
                server_model.eval()
                validation_batches_event.wait()
                validation_batches_event.clear()
                with cursor_lock:
                    self.fetching_cursor.execute(f'SELECT validation_batches FROM validation WHERE client_id = ?', client_id)
                    select_event.set()
                    select_event.wait()
                    select_event.clear()
                    client_batches = self.fetching_cursor.fetchone()[0]
                    self.fetching_connection.commit()
                    self.fetching_cursor.close()
                    self.fetching_cursor = self.fetching_connection.cursor()
                running_vloss = 0.0
                for batch in range(client_batches):
                    validation_outputs_event.wait()
                    validation_outputs_event.clear()
                    with cursor_lock:
                        self.fetching_cursor.execute(f'SELECT validation_outputs FROM validation WHERE client_id = ?', client_id)
                        select_event.set()
                        select_event.wait()
                        select_event.clear()
                        client_outputs = pickle.loads(self.fetching_cursor.fetchone()[0])
                        self.fetching_connection.commit()
                    self.fetching_cursor.close()
                    self.fetching_cursor = self.fetching_connection.cursor()
                    validation_labels_event.wait()
                    validation_labels_event.clear()
                    with cursor_lock:
                        self.fetching_cursor.execute(f'SELECT validation_labels FROM validation WHERE client_id = ?', client_id)
                        select_event.set()
                        select_event.wait()
                        select_event.clear()
                        validation_labels = pickle.loads(self.fetching_cursor.fetchone()[0])
                        self.fetching_connection.commit()
                        self.fetching_cursor.close()
                        self.fetching_cursor = self.fetching_connection.cursor()
                    voutputs1 = server_model(client_outputs)
                    vloss = loss_fn(voutputs1, validation_labels)
                    clientsocket.send(b'<OK>')
                    running_vloss += vloss


            
            avg_vloss = running_vloss / (batch + 1)
            print(f'Global round {epoch + 1} Client: {client_address}\n   Average Training Loss: {avg_loss: .3f}\n   Average Validation Loss: {avg_vloss: .3f}\n')
            self.client_trained_counter += 1
            with cursor_lock:
                client_cursor.execute('UPDATE weights SET server_updated_weights = ? WHERE client_id = ?', pickle.dumps(server_model.state_dict()), client_id)
                client_connection.commit()
            # Aggregate models when you have trained with all client for one global round
            if self.client_trained_counter == len(self.client_ids):
                weights = {}
                self.client_trained_counter = 0
                with cursor_lock:
                    self.aggregate_models(client_id, client_cursor, client_connection)


            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                #torch.save(model.state_dict(), model_path)
        #training_event.set()


class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
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
    
    

if __name__ == '__main__':
    Server()