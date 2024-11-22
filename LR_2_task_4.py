import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv('income_data.txt', delimiter=',')

# Визначення ознак і цільової змінної
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Попередня обробка даних
# Визначення категоріальних і числових змінних
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Кодування категоріальних даних та обробка пропущених значень
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Масштабування числових даних
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Використовуємо щільний формат
        ]), categorical_cols)
    ])

# Розділення даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список моделей
models = [
    ('Logistic Regression (LR)', LogisticRegression(max_iter=1000)),
    ('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()),
    ('K-Nearest Neighbors (KNN)', KNeighborsClassifier()),
    ('Classification and Regression Tree (CART)', DecisionTreeClassifier()),
    ('Naive Bayes (NB)', GaussianNB()),
    ('Support Vector Machine (SVM)', SVC())
]

# Оцінка моделей
results = []
names = []
for name, model in models:
    # Створення пайплайну для кожної моделі
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    results.append(cv_scores)
    names.append(name)
    print(f'{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# Візуалізація результатів
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot(results, tick_labels=names, showmeans=True)
plt.title('Comparison of Classification Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()
