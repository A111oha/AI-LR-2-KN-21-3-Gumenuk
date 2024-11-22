import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Завантаження даних
input_file = 'income_data.txt'

print("Читання даних з файлу...")

# Читання даних
X = []
y = []

try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if '?' in line:
                continue
            data = line[:-1].split(', ')  # розділення рядка по комі
            if len(data) == 15:  # перевірка чи є 15 елементів (за кількістю ознак + 1 для мітки класу)
                X.append(data[:-1])  # всі ознаки без останньої
                y.append(data[-1])   # мітка класу
            else:
                print(f"Пропуск рядка з неповними даними: {line}")
except FileNotFoundError:
    print(f"Файл {input_file} не знайдено.")
    exit()

print(f"Дані успішно завантажено: {len(X)} записів.")

# Перетворення на масив numpy
try:
    X = np.array(X)
    y = np.array(y)
except ValueError as e:
    print(f"Помилка при перетворенні в масив: {e}")
    exit()

print("Кодування текстових даних...")
# Кодування текстових даних
label_encoders = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder = LabelEncoder()
    X_encoded[:, i] = label_encoder.fit_transform(X[:, i])
    label_encoders.append(label_encoder)

# Масштабування ознак
scaler = StandardScaler()
X_encoded = scaler.fit_transform(X_encoded)

# Кодування міток
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

print("Кодування завершено.")

# Розділення на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=5)

print(f"Навчальний набір: {X_train.shape}, Тестовий набір: {X_test.shape}")

# Навчання з поліноміальним ядром
print("Навчання з поліноміальним ядром...")
classifier_poly = SVC(kernel='poly', degree=8, random_state=5)  # Видалено n_jobs
classifier_poly.fit(X_train, y_train)
y_pred_poly = classifier_poly.predict(X_test)

# Навчання з гаусовим ядром
print("Навчання з гаусовим ядром...")
classifier_rbf = SVC(kernel='rbf', random_state=5)  # Видалено n_jobs
classifier_rbf.fit(X_train, y_train)
y_pred_rbf = classifier_rbf.predict(X_test)

# Навчання з сигмоїдальним ядром
print("Навчання з сигмоїдальним ядром...")
classifier_sigmoid = SVC(kernel='sigmoid', random_state=5)  # Видалено n_jobs
classifier_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = classifier_sigmoid.predict(X_test)

# Оцінка якості
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Оцінка для кожного ядра
print("\nОцінка для поліноміального ядра:")
accuracy_poly, precision_poly, recall_poly, f1_poly = evaluate_model(y_test, y_pred_poly)
print(f"Accuracy: {accuracy_poly:.2f}, Precision: {precision_poly:.2f}, Recall: {recall_poly:.2f}, F1: {f1_poly:.2f}")

print("\nОцінка для гаусового ядра:")
accuracy_rbf, precision_rbf, recall_rbf, f1_rbf = evaluate_model(y_test, y_pred_rbf)
print(f"Accuracy: {accuracy_rbf:.2f}, Precision: {precision_rbf:.2f}, Recall: {recall_rbf:.2f}, F1: {f1_rbf:.2f}")

print("\nОцінка для сигмоїдального ядра:")
accuracy_sigmoid, precision_sigmoid, recall_sigmoid, f1_sigmoid = evaluate_model(y_test, y_pred_sigmoid)
print(f"Accuracy: {accuracy_sigmoid:.2f}, Precision: {precision_sigmoid:.2f}, Recall: {recall_sigmoid:.2f}, F1: {f1_sigmoid:.2f}")
