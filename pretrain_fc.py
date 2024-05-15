import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from dataloader.decentral_dataloader import DecentralPlannerDataLoader
from config.config_train import config_train

class SimpleFCNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=2, output_size=5):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self._initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

model = SimpleFCNN()
model = model.cuda()
model.train()
dataloader = DecentralPlannerDataLoader(config_train)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
optimizer.zero_grad()
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    k = 0
    s = 0
    while k < 10:
        for batch_idx, (batch_input, batch_target, _, batch_GSO, map) in enumerate(dataloader.train_loader):
            loss = 0
            total_matched_rows = 0
            input = batch_input[1].cuda()
            target = batch_target.cuda()
            optimizer.zero_grad()
            output = model(input)
            for agent in range(len(output)):
                loss += criterion(output[agent], torch.max(target[agent], 1)[1])
                max_indices = torch.argmax(output[agent], dim=1)
                one_indices = torch.nonzero(target[agent], as_tuple=False)
                matched_indices = torch.eq(max_indices, one_indices[:, 1])
                total_matched_rows += torch.sum(matched_indices).item()
            loss.backward()
            if total_matched_rows > 500 and loss.item() < 48 and s == 0:
                with open('save.pkl', 'wb') as f:
                    pickle.dump(model.state_dict(), f)
                s = 1
                best_matched_rows = total_matched_rows
                best_loss = loss.item()
            if s == 1 and total_matched_rows > best_matched_rows and loss.item() < best_loss:
                best_matched_rows = total_matched_rows
                best_loss = loss.item()
                with open('save.pkl', 'wb') as f:
                    pickle.dump(model.state_dict(), f)
            optimizer.step()
            print(f'loss: {loss.item()}')
            print(f"-------Accuracy is {total_matched_rows}/640----------")
