import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:  # Пропускаємо рядки з невідомими даними
            continue

        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)  # Клас для <=50K
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)  # Клас для >50K
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)
y = np.array(y)

# Перетворення категоріальних ознак у числові
label_encoders = []
X_encoded = np.empty(X.shape)

for i in range(X.shape[1]):
    if X[0, i].isdigit():  # Якщо числове
        X_encoded[:, i] = X[:, i].astype(float)
    else:  # Якщо текстове
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

X = X_encoded.astype(float)
# Розбиття на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Навчання класифікатора
classifier.fit(X_train, y_train)
# Прогнозування результатів для тестового набору
y_test_pred = classifier.predict(X_test)

# Обчислення F1-міри
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1 score: {round(100 * f1, 2)}%")
# Вхідні дані
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки
input_data_encoded = np.empty(len(input_data))
category_index = 0  # Лічильник для категоріальних ознак

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)  # Числові дані залишаємо без змін
    else:
        # Обробка категоріальних даних
        if item in label_encoders[category_index].classes_:
            input_data_encoded[i] = label_encoders[category_index].transform([item])[0]
        else:
            # Якщо значення невідоме, додаємо його до класів
            new_classes = np.append(label_encoders[category_index].classes_, item)
            label_encoders[category_index].classes_ = new_classes
            input_data_encoded[i] = label_encoders[category_index].transform([item])[0]
        category_index += 1  # Збільшуємо лічильник для категоріальних атрибутів
# Прогнозування
input_data_encoded = input_data_encoded.reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)
result = '<=50K' if predicted_class[0] == 0 else '>50K'
print(f"Predicted class for the input data: {result}")
y_test_pred = classifier.predict(X_test)
# Обчислення метрик
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
# Вивід результатів
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

