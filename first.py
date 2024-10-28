import torch
import torch.nn as nn
import torch.optim as optim

# Определяем данные для задачи XOR
inputs = torch.tensor([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]], dtype=torch.float32)

targets = torch.tensor([[0],
                        [1],
                        [1],
                        [0],
                        [1],
                        [0],
                        [0],
                        [1]], dtype=torch.float32)

# Определяем архитектуру нейросети
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(3, 2)  # Входной слой с 3 нейронами и скрытый слой с 6 нейронами
        self.output = nn.Linear(2, 1)  # Выходной слой с 1 нейроном

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # ReLU для ускорения сходимости
        x = torch.sigmoid(self.output(x))  # Сигмоида на выходном слое для предсказания
        return x

# Инициализируем модель, функцию потерь и оптимизатор (Adam вместо SGD)
model = XORNet()
criterion = nn.MSELoss()  # Среднеквадратичная ошибка для задачи регрессии
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam для ускорения обучения

# Функция обучения нейросети
def train_model(model, inputs, targets, target_accuracy, max_epochs=100000):
    for epoch in range(max_epochs):
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Вычисляем среднюю ошибку по всем примерам
        avg_error = torch.mean(torch.abs(outputs - targets))

        if avg_error.item() <= target_accuracy:
            print(f'Точность достигнута на {epoch + 1}-й эпохе с ошибкой: {avg_error.item():.5f}')
            break

        # Печатаем каждые 1000 эпох
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.5f}, Error: {avg_error.item():.5f}')

# Обучаем модель до заданной точности
target_accuracy = 0.001  # Точность: средняя ошибка не больше 0.005
train_model(model, inputs, targets, target_accuracy)

# Тестируем модель после обучения
with torch.no_grad():  # Отключаем вычисление градиентов для теста
    test_outputs = model(inputs)
    print("\nРезультаты на обучающем наборе:")
    for i in range(len(inputs)):
        print(f'Вход: {inputs[i].numpy()}, Предсказание: {test_outputs[i].item():.5f}, Истинное значение: {targets[i].item()}')
