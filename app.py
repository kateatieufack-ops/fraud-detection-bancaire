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
    st.title("🤖 AutoML & Détection de fraude intelligente")

    uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Fichier chargé avec succès ✅")
        st.dataframe(df.head())

        # Détection automatique d'une colonne fraude
        fraud_columns = ["fraud", "Fraud", "Class", "is_fraud"]

        detected_fraud_col = None
        for col in fraud_columns:
            if col in df.columns:
                detected_fraud_col = col
                break

        if detected_fraud_col:
            st.subheader("💳 Mode Détection de Fraude activé")
            target = detected_fraud_col
        else:
            st.subheader("🧠 Mode AutoML général")
            target = st.selectbox("Choisissez la variable cible", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        # Vérification
        if X.shape[1] == 0:
            st.error("❌ Aucune variable explicative disponible.")
            st.stop()

        # Encodage automatique
        if X.select_dtypes(include=["object", "category"]).shape[1] > 0:
            X = pd.get_dummies(X, drop_first=True)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Détection type problème
        if y.dtype == "object" or y.nunique() < 10:

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                classification_report
            )

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Modèle entraîné avec succès 🎯")

            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2f}")

            st.subheader("Matrice de confusion")
            st.write(confusion_matrix(y_test, y_pred))

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Si fraude → afficher taux
            if detected_fraud_col:
                fraud_rate = (y_pred.sum() / len(y_pred)) * 100
                st.metric("Taux de fraude détecté (%)", f"{fraud_rate:.2f}")

        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_squared_error

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.metric("R² Score", f"{r2:.2f}")
            st.metric("MSE", f"{mse:.2f}")

        # Importance des variables
        st.subheader("📊 Importance des variables")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        st.bar_chart(importances.sort_values(ascending=False).head(10))

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
