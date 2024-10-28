import torch
import torch.nn as nn
import random

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
        self.hidden = nn.Linear(3, 3)  # Входной слой с 3 нейронами и скрытый слой с 3 нейронами
        self.output = nn.Linear(3, 1)  # Выходной слой с 1 нейроном

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # ReLU для скрытого слоя
        x = torch.sigmoid(self.output(x))  # Сигмоида на выходе для вероятности
        return x

# Функция для оценки приспособленности
def fitness(model, inputs, targets):
    with torch.no_grad():
        outputs = model(inputs)
        mse = torch.mean((outputs - targets) ** 2)  # Среднеквадратичная ошибка
    return mse.item()

# Функция для скрещивания двух сетей с вероятностью выбора от одного из родителей
def crossover(parent1, parent2):
    child = XORNet()
    with torch.no_grad():
        for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.randint(0, 2, p1_param.shape).float()  # Случайная маска
            child_param.data.copy_(mask * p1_param.data + (1 - mask) * p2_param.data)  # Выбор весов от одного из родителей
    return child

# Функция для мутации сети с мягким воздействием
def mutate(model, mutation_rate):
    with torch.no_grad():
        for param in model.parameters():
            if random.random() < mutation_rate:

                noise = torch.randn_like(param)
                param.add_(noise)

# Инициализируем популяцию из 100 нейросетей
population_size = 100
population = [XORNet() for _ in range(population_size)]

# Эволюционные параметры
num_generations = 500  # Количество поколений
survival_rate = 0.5  # Оставляем 50% лучших сетей
mutation_rate = 0.2  # Вероятность мутации

# Моделирование эволюции
for generation in range(num_generations):
    # Оценка приспособленности всех сетей
    fitness_scores = [(model, fitness(model, inputs, targets)) for model in population]
    fitness_scores.sort(key=lambda x: x[1])  # Сортируем по возрастанию ошибки

    # Отбираем лучших для размножения
    num_survivors = int(population_size * survival_rate)
    survivors = [model for model, _ in fitness_scores[:num_survivors]]

    # Скрещиваем лучших и создаем новое поколение
    next_generation = survivors[:]
    while len(next_generation) < population_size:
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate)
        next_generation.append(child)

    # Обновляем популяцию
    population = next_generation

    # Выводим лучшую сеть в поколении
    best_fitness = fitness_scores[0][1]
    print(f'Поколение {generation + 1}: Лучшая приспособленность = {best_fitness:.6f}')

# Тестирование лучшей сети после эволюции
best_model = fitness_scores[0][0]
with torch.no_grad():
    test_outputs = best_model(inputs)
    print("\nРезультаты на обучающем наборе для лучшей модели:")
    for i in range(len(inputs)):
        print(f'Вход: {inputs[i].numpy()}, Предсказание: {test_outputs[i].item():.5f}, Истинное значение: {targets[i].item()}')
