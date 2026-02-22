import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split

# -------------------------------
# Charger le modèle
# -------------------------------
model = joblib.load("fraud_model.pkl")  # modèle Random Forest déjà entraîné

# Liste complète des colonnes que le modèle attend
model_features = ["Time", "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                  "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                  "V21","V22","V23","V24","V25","V26","V27","V28","Amount"]

# -------------------------------
# Barre de navigation
# -------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Choisissez une page :", 
                        ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse des données",
                         "📈 Performance du modèle", "📂 Prédiction par fichier", "ℹ️ À propos"])

# -------------------------------
# ACCUEIL
# -------------------------------
if menu == "🏠 Accueil":
    st.title("🔍 Détection de Fraude par Carte de Crédit")
    st.markdown("""
    Bienvenue sur l'application de Détection de Fraude !

    📖 **À propos du projet**  
    Cette application utilise le Machine Learning pour détecter les transactions frauduleuses par carte de crédit.  
    **Le principal défi :** le déséquilibre marqué des classes (moins de 6% de fraudes réelles) et la détection précise de comportements atypiques dans des millions de transactions.

    🎯 **Fonctionnalités principales :**
    - ✅ Prédiction individuelle en temps réel
    - 📊 Exploration interactive des données
    - 📈 Métriques complètes (F1, AUC-ROC, Rappel)
    - 📂 Prédiction en masse sur n'importe quel fichier CSV
    - 🤖 Modèle Random Forest optimisé par GridSearchCV
    - 🎨 Visualisations avancées pour mieux comprendre les transactions
    - ⚡ Interface intuitive et professionnelle
    - 🔧 Adaptation automatique aux colonnes manquantes pour import CSV
    """)

# -------------------------------
# PRÉDICTION INDIVIDUELLE
# -------------------------------
elif menu == "🔮 Prédiction":
    st.title("💳 Prédiction sur une Transaction")
    st.write("Entrez les caractéristiques d'une transaction pour obtenir une prédiction :")

    montant = st.number_input("💰 Montant (€)", value=300.0)
    temps = st.number_input("⏱️ Temps (secondes)", value=10000.0)

    V = {}
    for i in range(1,29):
        V[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    if st.button("🎯 Analyser la prédiction"):
        df_input = pd.DataFrame(columns=model_features)
        df_input.loc[0, "Time"] = temps
        df_input.loc[0, "Amount"] = montant
        for i in range(1,29):
            df_input.loc[0, f"V{i}"] = V[f"V{i}"]
        df_input = df_input.fillna(0)

        try:
            prediction = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1]
            st.markdown(f"**Résultat :** {'💥 Fraude détectée !' if prediction==1 else '✅ Transaction légitime'}")
            st.markdown(f"**Probabilité de fraude :** {proba*100:.2f}%")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# -------------------------------
# ANALYSE DES DONNÉES
# -------------------------------
elif menu == "📊 Analyse des données":
    st.title("📊 Exploration et Visualisation des Données")
    df = pd.read_csv("sample_creditcard.csv")
    st.write("Aperçu des données :")
    st.dataframe(df.head())

    total_trans = len(df)
    total_fraudes = df["Class"].sum()
    taux_fraude = total_fraudes / total_trans * 100
    variables = df.shape[1]-1

    st.metric("Total transactions", total_trans)
    st.metric("Fraudes", total_fraudes)
    st.metric("Taux de fraude (%)", f"{taux_fraude:.2f}")
    st.metric("Variables", variables)

    st.subheader("Distribution des classes")
    plt.figure(figsize=(6,4))
    sns.countplot(x="Class", data=df)
    st.pyplot(plt)

    st.subheader("Répartition des montants")
    plt.figure(figsize=(6,4))
    sns.histplot(df["Amount"], bins=50, kde=True)
    st.pyplot(plt)

    st.subheader("Matrice de corrélation")
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

# -------------------------------
# PERFORMANCE DU MODÈLE
# -------------------------------
elif menu == "📈 Performance du modèle":
    st.title("📈 Performance du Modèle")
    df = pd.read_csv("sample_creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    st.pyplot(plt)

    st.subheader("Courbe ROC")
    RocCurveDisplay.from_predictions(y_test, y_proba)
    st.pyplot(plt)

    st.write("💡 Notes sur la performance :")
    st.write("""
    - F1-score élevé (~0.95) : très bon compromis précision/recall
    - AUC-ROC proche de 1 : excellent pour le déséquilibre
    - Risque d'erreur faible mais présent : certaines fraudes peuvent être manquées
    """)

# -------------------------------
# PRÉDICTION PAR FICHIER
# -------------------------------
elif menu == "📂 Prédiction par fichier":
    st.title("📁 Prédiction en masse — Import CSV universel")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    if uploaded_file:
        df_file = pd.read_csv(uploaded_file)
        st.write("Aperçu du fichier importé :")
        st.dataframe(df_file.head())

        for col in model_features:
            if col not in df_file.columns:
                df_file[col] = 0
        df_file = df_file[model_features]

        predictions = model.predict(df_file)
        df_file["Fraude"] = predictions
        st.write("Résultats de la détection :")
        st.dataframe(df_file)

# -------------------------------
# À PROPOS
# -------------------------------
elif menu == "ℹ️ À propos":
    st.title("ℹ️ À propos")
    st.markdown("""
    **ATIEUFACK GUETSOP SHERONNE KATE**  
    🎓 Partie 3 — TP2 IIA | LICENCE MTQ S6 | IUSJ Cameroun 2025-2026  
    Par Stéphane C. K. TÉKOUABOU (PhD & Ing.)  

    🛠️ **Technologies :** Python, Scikit-learn, Pandas & NumPy, Matplotlib & Seaborn, Streamlit, Joblib  
    🤖 **Modèle :** Random Forest Classifier optimisé par GridSearchCV  
    Gestion du déséquilibre par Oversampling  

    💡 Cette application permet de détecter efficacement les fraudes sur les cartes de crédit, d’explorer les données et d’analyser la performance du modèle.
    """)
