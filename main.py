import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据集转化和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc_out = nn.Linear(512, 10)
        self.fc_select = nn.Linear(512, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        out = self.fc_out(x)
        selection = torch.sigmoid(self.fc_select(x))
        return out, selection


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, selection = model(inputs)

            loss_main = criterion(outputs, labels)

            selected_outputs = outputs * selection
            selected_labels = labels.clone().detach()

            valid_indices = (selection > 0.5).nonzero(as_tuple=True)[1]
            filtered_outputs = selected_outputs[valid_indices]
            filtered_labels = selected_labels[valid_indices]

            loss_select = criterion(filtered_outputs, filtered_labels)

            loss = loss_main + loss_select
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



def test_model():
    model.eval()
    correct_main = 0
    correct_selected = 0
    total_main = 0
    total_selected = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, selection = model(inputs)


            _, predicted_main = torch.max(outputs.data, 1)
            correct_main += (predicted_main == labels).sum().item()
            total_main += labels.size(0)


            selection = (selection > 0.5).float()
            selected_outputs = outputs * selection.unsqueeze(1)  # 保留输出
            valid_indices = (selection > 0.5).nonzero(as_tuple=True)[0]

            if valid_indices.numel() > 0:
                filtered_outputs = selected_outputs[valid_indices]
                filtered_labels = labels[valid_indices]


                if filtered_labels.numel() > 0:
                    _, predicted_selected = torch.max(filtered_outputs.data, 1)
                    correct_selected += (predicted_selected == filtered_labels).sum().item()
                    total_selected += filtered_labels.size(0)  # 计算实际筛选的样本数

    print(f'Accuracy of the original model: {100 * correct_main / total_main:.2f}%')
    if total_selected > 0:
        print(f'Accuracy of the retained model: {100 * correct_selected / total_selected:.2f}%')
    else:
        print('No valid samples to calculate the retained accuracy.')

train_model(num_epochs=10)
test_model()