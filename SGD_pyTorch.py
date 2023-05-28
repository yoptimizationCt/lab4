import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

eps_point = [20, 20]
modes = {'SGD': 0.0009, 'Momentum': 0.0009, 'Nesterov': 0.0009, 'Adam': 1, 'AdaGrad': 1, 'RMSProp': 0.1}


def array_to_torch(A):
    return torch.from_numpy(np.array([[a] for a in A])).float()


def SGD_PT(X, Y, lr, mode='SGD', epochs=100, log=False, initial_guess=None):
    # Модель линейной регрессии
    # Если взять, что у нас Y = wX + b, то w = weight в нашей модели, а b - bias
    if initial_guess is None:
        initial_guess = [0, 0]
    points = np.zeros((epochs + 1, 2))
    model = nn.Linear(1, 1)
    model.weight.data = torch.tensor([[float(initial_guess[0])]])
    model.bias.data = torch.tensor([[float(initial_guess[1])]])
    # Функция потерь
    criterion = nn.MSELoss()
    optimizer = ''
    # Оптимизатор и скорость обучения
    # Ordinary SGD
    if mode == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    # SGD with momentum
    elif mode == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # SGD Nesterov
    elif mode == 'Nesterov':
        optimizer = optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    # AdaGrad
    elif mode == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # RMSProp
    elif mode == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # Adam
    elif mode == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epoch = epochs
    for epoch in range(epochs):
        # Подсчитываем ошибку модели
        outputs = model(X)
        loss = criterion(outputs, Y)

        # Обнуляем градиент, если мы не хотим накапливать градиент
        optimizer.zero_grad()
        # Обновляем градиент
        loss.backward()

        gradient_values = [param.grad.item() for param in model.parameters()]
        if abs(gradient_values[0]) < eps_point[0] and abs(gradient_values[1]) < eps_point[1]:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
            print('Predicted:', model.weight.item(), model.bias.item())
            num_epoch = epoch + 1
            break
        # Делаем шаг градиентного спуска
        optimizer.step()
        points[epoch + 1] = np.array([model.weight.item(), model.bias.item()])
        if log:
            # Выводим промежуточные результаты
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
                print('Predicted:', model.weight.item(), model.bias.item())
    return points, num_epoch


n = 1000
X = np.random.rand(n) * 10
# epoch from number of batch
number_of_epochs = {}
epoch_count = 500
batch_number = 30
for a_exp in [1, 3, 5, 7, 10]:
    for noise in [1, 5, 10, 20, 30]:
        Y = a_exp * X + np.random.rand(n) * noise
        funcStr = str(a_exp) + "x_noise(" + str(noise) + ")"
        for mode in modes:
            print(funcStr)
            print("ЕБАТЬ ДА ЭТО ЖЕ ЧЕРТОВ", mode)
            points = SGD_PT(array_to_torch(X),
                            array_to_torch(Y), modes[mode], mode=mode, epochs=1000)
            print(points[1])
            print(points[0][:points[1]])
            print('-----------')
