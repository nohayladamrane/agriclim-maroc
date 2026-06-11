"""
Script pour sauvegarder le modèle XGBoost
Exécutez ce script UNE SEULE FOIS avant de lancer l'application
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost as xgb

print("=" * 60)
print("SAUVEGARDE DU MODÈLE XGBOOST - AgriClim Maroc")
print("=" * 60)

# 1. Vérifier que les données existent
if not os.path.exists("dataset_ml.csv"):
    print("\n❌ ERREUR: Fichier dataset_ml.csv introuvable !")
    print("   Assurez-vous d'avoir d'abord exécuté votre CODE 5")
    print("   pour générer les datasets.")
    exit(1)

# 2. Charger les données
print("\n📂 Chargement des données...")
df = pd.read_csv("dataset_ml.csv", encoding='utf-8-sig')
print(f"   ✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 3. Identifier les colonnes disponibles
feature_cols = ['Precip_Total_mm', 'Temp_Mean_C', 'Humidity_Pct', 'Solar_Radiation', 'Year']
available_features = [f for f in feature_cols if f in df.columns]

if 'Crop_Type' in df.columns:
    available_features.append('Crop_Type')
    print(f"   + Feature catégorielle: Crop_Type")

print(f"\n📊 Features utilisées: {available_features}")

# 4. Préparer les données
X = df[available_features].copy()
y = df['Value_Mean'].copy()
y_log = np.log1p(y)  # Transformation log

# 5. Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)
print(f"\n📊 Split données: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# 6. Créer le préprocesseur
numeric_features = [f for f in available_features if f != 'Crop_Type']

if 'Crop_Type' in available_features:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Crop_Type'])
    ])
else:
    preprocessor = StandardScaler()

# 7. Créer le modèle XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# 8. Créer le pipeline complet
if 'Crop_Type' in available_features:
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', xgb_model)])
else:
    pipeline = Pipeline([('scaler', preprocessor), ('regressor', xgb_model)])

# 9. Entraîner le modèle
print("\n🔄 Entraînement du modèle XGBoost en cours...")
pipeline.fit(X_train, y_train)

# 10. Évaluer le modèle
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
r2 = r2_score(y_test_orig, y_pred)

print(f"\n📈 Performance du modèle: R² = {r2:.4f}")
if r2 >= 0.92:
    print("   ✅ Excellente performance (conforme au rapport)")
else:
    print(f"   ⚠️ Performance inférieure aux 0.9259 du rapport")

# 11. Créer le dossier models
os.makedirs("models", exist_ok=True)

# 12. Sauvegarder le modèle
joblib.dump(pipeline, "models/xgboost_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\n" + "=" * 60)
print("✅ SUCCÈS - Modèle sauvegardé !")
print("=" * 60)
print("\n📁 Fichiers créés:")
print("   - models/xgboost_model.pkl")
print("   - models/preprocessor.pkl")
print("\n👉 Vous pouvez maintenant lancer l'application Streamlit:")
print("   streamlit run app.py")