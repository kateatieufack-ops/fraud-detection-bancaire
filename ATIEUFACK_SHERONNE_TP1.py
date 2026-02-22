import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

# Charger les données
df = pd.read_csv("creditcard.csv")

# Aperçu des données
print(df.head())
print("Shape :", df.shape)
print(df.isnull().sum())
print("Distribution des classes :\n", df['Class'].value_counts())

# Graphique 1 : Distribution des classes
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Distribution des classes (0 = Légitime, 1 = Fraude)")
plt.savefig("class_distribution.png")  # sauvegarde pour Streamlit
plt.show()

# Graphique 2 : Répartition des montants
plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Répartition des montants des transactions")
plt.savefig("amount_distribution.png")
plt.show()

# Graphique 3 : Matrice de corrélation
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.savefig("correlation_matrix.png")
plt.show()

# Préparation des données pour le modèle
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# Sauvegarde du modèle pour Streamlit
joblib.dump(model, "fraud_model.pkl")
print("Modèle sauvegardé avec succès.")