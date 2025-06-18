import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import joblib

# 1 Carregar os dados
df = pd.read_csv("data/asl_dataset.csv")  # precisa ter a coluna 'label'

X = df.drop("label", axis=1)
y = df["label"]

# 2   Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#3. Treinar RandomForest com GridSearchCV
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "max_features": ["sqrt"]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_params, cv=StratifiedKFold(n_splits=5), scoring="f1_macro", verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# 4   Ttreinar lightGBM com gridSearchCV
lgb_params = {
    "n_estimators": [100, 200],
    "num_leaves": [31, 64],
    "learning_rate": [0.1, 0.01]
}
lgb = LGBMClassifier(random_state=42)
grid_lgb = GridSearchCV(lgb, lgb_params, cv=StratifiedKFold(n_splits=5), scoring="f1_macro", verbose=1, n_jobs=-1)
grid_lgb.fit(X_train, y_train)

# 5 avaliar os modeloss
print(" RandomForest Report:")
print(classification_report(y_test, grid_rf.predict(X_test)))
print(confusion_matrix(y_test, grid_rf.predict(X_test)))

print("\n LightGBM Report:")
print(classification_report(y_test, grid_lgb.predict(X_test)))
print(confusion_matrix(y_test, grid_lgb.predict(X_test)))

# 6 salvar os melhores modelos
joblib.dump(grid_rf.best_estimator_, "models/random_forest_model.pkl")
joblib.dump(grid_lgb.best_estimator_, "models/lightgbm_model.pkl")

print(" Modelos treinados e salvos com sucesso.")
