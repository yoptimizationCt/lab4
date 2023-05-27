import torch
import torch.nn as nn
import torch.optim as optim

# Входные данные
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Модель линейной регрессии
# Если взять, что у нас Y = wX + b, то w = weight в нашей модели, а b - bias
model = nn.Linear(1, 1)

# Функция потерь
criterion = nn.MSELoss(reduction='sum')

# Оптимизатор и скорость обучения
# Ordinary SGD
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# SGD with momentum
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# SGD Nesterov
# optimizer = optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9)
# AdaGrad
# optimizer = optim.Adagrad(model.parameters())
# RMSProp
# optimizer = optim.RMSprop(model.parameters())
# Adam
optimizer = optim.Adam(model.parameters())

# Количество эпох
num_epochs = 100

for epoch in range(num_epochs):
    # Подсчитываем ошибку модели
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Обнуляем градиент, если мы не хотим накапливать градиент
    optimizer.zero_grad()
    # Обновляем градиент
    loss.backward()
    # Делаем шаг градиентного спуска
    optimizer.step()

    # Выводим промежуточные результаты
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print('Predicted:', model.weight.item(), model.bias.item())

# Чекаем результаты
predicted = model(X)
print('Predicted:', model.weight.item(), model.bias.item())
print('Predicted:', predicted)
