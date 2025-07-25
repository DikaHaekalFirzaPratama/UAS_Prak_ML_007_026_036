import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('StudentsPerformance.csv')

# Buat label target: math score tinggi (>=70)
df['math_target'] = df['math score'].apply(lambda x: 1 if x >= 70 else 0)
df = df.drop(['math score'], axis=1)

# Encode fitur kategorikal
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Fitur dan target
X = df.drop('math_target', axis=1)
y = df['math_target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluasi
print("=== K-Nearest Neighbors ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
