import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris_dataset = load_iris()

# Перегляд основної інформації про набір даних
print("Ключі iris_dataset: \n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: ", iris_dataset['target_names'])
print("Назва ознак: \n", iris_dataset['feature_names'])
print("Тип масиву data: ", type(iris_dataset['data']))
print("Форма масиву data: ", iris_dataset['data'].shape)
print("Значення ознак перших 5 прикладів:\n", iris_dataset['data'][:5])
print("Тип масиву target: ", type(iris_dataset['target']))
print("Мітки:\n", iris_dataset['target'])

# 2. Завантаження даних через pandas (альтернативний спосіб)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Огляд даних
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# 3. Візуалізація даних
# Діаграми розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Гістограми розподілу
dataset.hist()
plt.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# 4. Розподіл даних на навчальний і тестовий набори
X = dataset.iloc[:, 0:4].values  # Вибір ознак
y = dataset.iloc[:, 4].values  # Вибір класів
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Перевірка форм
print("Форма X_train:", X_train.shape)
print("Форма X_test:", X_test.shape)

# 5. Створення та оцінка моделі
# Модель логістичної регресії
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка
print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred))
print("Класифікаційний звіт:\n", classification_report(y_test, y_pred))
print("Точність:", accuracy_score(y_test, y_pred))