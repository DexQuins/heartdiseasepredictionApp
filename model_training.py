import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_save_model():
    data = pd.read_csv("heart_disease_data.csv")

    print("\nKolom dataset:", data.columns)
    print("\nDistribusi target:")
    print(data['target'].value_counts())

    # 🩹 JIKA CUMA 1 KELAS
    if data['target'].nunique() < 2:
        print("⚠️ Target cuma 1 kelas, nambah dummy data")
        dummy = data.iloc[0:1].copy()
        dummy['target'] = 1
        data = pd.concat([data, dummy], ignore_index=True)

    X = data.drop("target", axis=1)
    y = data["target"]

    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🚀 Mulai training...")
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"✅ Accuracy: {acc:.2f}")

    joblib.dump(pipeline, "heart_disease_model.pkl")
    joblib.dump(X.columns.tolist(), "feature_names.pkl")

    print("💾 Model & fitur berhasil disimpan")

if __name__ == "__main__":
    train_and_save_model()
