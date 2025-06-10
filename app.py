# ===============================================
# 🧠 STREAMLIT APP: Prediksi Customer Churn (Final, Lengkap, Estetis)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import datetime

# ------------------------------------------
# 🛠️ Konfigurasi Halaman Utama
# ------------------------------------------
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-box {
    background-color: #2a2b3b;
    padding: 20px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# 📌 Sidebar Navigasi
# ------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1055/1055646.png", width=80)
st.sidebar.title("🔍 Aplikasi Prediksi Churn")
st.sidebar.markdown("""
Aplikasi ini memprediksi kemungkinan pelanggan berhenti langganan (**churn**) berdasarkan histori layanan.
""")
menu = st.sidebar.radio("Navigasi:", [
    "📌 Dashboard", 
    "📊 Visualisasi & Evaluasi", 
    "🔮 Prediksi Pelanggan"])

# ------------------------------------------
# 📥 Load Dataset dan Preprocessing
# ------------------------------------------
@st.cache_data

def load_data():
    df = pd.read_csv("data-customer-churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

df = load_data()

@st.cache_data
def preprocess(df):
    df_clean = df.copy()
    df_clean.drop("customerID", axis=1, inplace=True)
    df_clean["Churn"] = df_clean["Churn"].map({"No": 0, "Yes": 1})
    df_clean = pd.get_dummies(df_clean)
    scaler = StandardScaler()
    df_clean[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
        df_clean[["tenure", "MonthlyCharges", "TotalCharges"]])
    return df_clean

data = preprocess(df)
X = data.drop("Churn", axis=1)
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------
# 📌 DASHBOARD
# ------------------------------------------
if menu == "📌 Dashboard":
    st.title("📌 Dashboard Ringkasan Dataset")
    st.success("✅ Data berhasil dimuat.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Jumlah Data</h3>
            <h1>{len(df)}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Jumlah Fitur</h3>
            <h1>{len(df.columns)}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-box'>
            <h3>Target</h3>
            <h1>Churn (Yes/No)</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Distribusi Churn")
    churn_count = df['Churn'].value_counts()
    churn_percent = (churn_count / len(df) * 100).round(2)
    churn_table = pd.DataFrame({
        'Kategori': ['Tidak Churn', 'Churn'],
        'Jumlah': churn_count.values,
        'Persentase': churn_percent.values
    })
    st.dataframe(churn_table)

    st.markdown("---")
    st.subheader("📊 Statistik Dataset Lengkap")

    with st.expander("🔹 Jenis Kelamin"):
        st.markdown("""
        - Laki-laki (Male): ±3.552 pelanggan  
        - Perempuan (Female): ±3.480 pelanggan
        """)

    with st.expander("🔹 Status Pasangan"):
        st.markdown("""
        - Sudah menikah (Yes): ±3.483 pelanggan  
        - Belum menikah (No): ±3.549 pelanggan
        """)

    with st.expander("🔹 Memiliki Tanggungan Anak"):
        st.markdown("""
        - Ya (Yes): ±2.113 pelanggan  
        - Tidak (No): ±4.919 pelanggan
        """)

    with st.expander("🔹 Jenis Kontrak"):
        st.markdown("""
        - Bulanan (Month-to-month): ±3.873 pelanggan  
        - Satu tahun (One year): ±1.471 pelanggan  
        - Dua tahun (Two year): ±1.688 pelanggan
        """)

    with st.expander("🔹 Metode Pembayaran"):
        st.markdown("""
        - Electronic check: ±2.369 pelanggan  
        - Mailed check: ±1.397 pelanggan  
        - Bank transfer (automatic): ±1.540 pelanggan  
        - Credit card (automatic): ±1.726 pelanggan
        """)

    with st.expander("🔹 Distribusi Churn"):
        st.markdown("""
        - Tidak churn (No): ±5.174 pelanggan (73.6%)  
        - Churn (Yes): ±1.858 pelanggan (26.4%)
        """)

    with st.expander("🔹 Biaya Bulanan & Total Biaya"):
        st.markdown("""
        **Biaya Bulanan (MonthlyCharges)**:  
        - Rata-rata: ±64.76  
        - Minimum: ±18.25  
        - Maksimum: ±118.75

        **Total Biaya (TotalCharges)**:  
        - Rata-rata: ±2.284,44  
        - Minimum: ±18.80  
        - Maksimum: ±8.684,80
        """)

    st.markdown("---")
    st.subheader("📋 Tentang Aplikasi & Dokumentasi")
    st.markdown("""
    Aplikasi ini bertujuan untuk memprediksi apakah pelanggan akan berhenti berlangganan (churn) atau tetap menggunakan layanan. Model yang digunakan adalah **Logistic Regression**.

    **Fitur utama yang digunakan untuk prediksi:**
    - **Tenure**: Lama berlangganan
    - **MonthlyCharges**: Tagihan per bulan
    - **TotalCharges**: Total tagihan
    - **Contract**: Jenis kontrak
    - **InternetService**: Jenis layanan internet
    """)

    st.markdown("---")

# ------------------------------------------
# 📊 VISUALISASI & EVALUASI
# ------------------------------------------
if menu == "📊 Visualisasi & Evaluasi":
    st.title("📊 Visualisasi dan Evaluasi Model")
    view = st.radio("Pilih Tampilan:", ["Visualisasi Distribusi", "Evaluasi Model"])

    if view == "Visualisasi Distribusi":
        fitur = st.selectbox("Pilih Fitur untuk Divisualisasikan:", [
            "Tenure", "Contract", "InternetService",
            "MonthlyCharges", "TotalCharges", "PaymentMethod",
            "gender", "SeniorCitizen", "Dependents"])

        fig, ax = plt.subplots(figsize=(5, 3))
        if fitur == "Tenure":
            sns.histplot(df, x="tenure", hue="Churn", multiple="stack", kde=True, ax=ax)
            caption = "Distribusi pelanggan berdasarkan lama berlangganan (tenure)."
        elif fitur == "Contract":
            sns.countplot(df, x="Contract", hue="Churn", ax=ax)
            caption = "Distribusi jenis kontrak dan dampaknya terhadap churn."
        elif fitur == "InternetService":
            sns.countplot(df, x="InternetService", hue="Churn", ax=ax)
            caption = "Distribusi jenis layanan internet terhadap churn."
        elif fitur == "MonthlyCharges":
            sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
            caption = "Tagihan bulanan vs kemungkinan churn."
        elif fitur == "TotalCharges":
            sns.boxplot(x="Churn", y="TotalCharges", data=df, ax=ax)
            caption = "Total tagihan vs churn pelanggan."
        elif fitur == "PaymentMethod":
            sns.countplot(x="PaymentMethod", hue="Churn", data=df, ax=ax)
            plt.xticks(rotation=45)
            caption = "Distribusi metode pembayaran terhadap churn."
        elif fitur == "gender":
            sns.countplot(x="gender", hue="Churn", data=df, ax=ax)
            caption = "Distribusi churn berdasarkan jenis kelamin."
        elif fitur == "SeniorCitizen":
            sns.countplot(x="SeniorCitizen", hue="Churn", data=df, ax=ax)
            caption = "Apakah pelanggan lansia lebih cenderung churn?"
        elif fitur == "Dependents":
            sns.countplot(x="Dependents", hue="Churn", data=df, ax=ax)
            caption = "Pengaruh memiliki tanggungan terhadap churn."

        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.markdown(f"**Keterangan**: {caption}")

    elif view == "Evaluasi Model":
        st.subheader("📉 Evaluasi Kinerja Model")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        col1, col2, col3 = st.columns(3)
        col1.metric("Akurasi", f"{acc:.2f}")
        col2.metric("Presisi", f"{prec:.2f}")
        col3.metric("Recall", f"{rec:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("F1 Score", f"{f1:.2f}")
        col5.metric("ROC AUC", f"{auc:.2f}")

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title("Confusion Matrix")

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_title("ROC Curve")
        ax2.legend()

        col6, col7 = st.columns(2)
        with col6:
            st.pyplot(fig1)
        with col7:
            st.pyplot(fig2)

        with st.expander("📋 Classification Report"):
            st.code(classification_report(y_test, y_pred))

# ------------------------------------------
# 🔮 PREDIKSI PELANGGAN
# ------------------------------------------
if menu == "🔮 Prediksi Pelanggan":
    st.title("🔮 Prediksi Churn Pelanggan")
    st.markdown("Masukkan detail pelanggan untuk memprediksi kemungkinan churn.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Lama Berlangganan (Tenure)", min_value=0, max_value=72, value=12)
            monthly = st.select_slider("Tagihan Bulanan", options=sorted(df["MonthlyCharges"].unique()))
            total = st.select_slider("Total Tagihan", options=sorted(df["TotalCharges"].unique()))
        with col2:
            contract = st.selectbox("Jenis Kontrak", df["Contract"].unique())
            internet = st.selectbox("Layanan Internet", df["InternetService"].unique())

        submit = st.form_submit_button("🔍 Prediksi")

    if submit:
        input_data = {
            "tenure": [tenure],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
            "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
            "Contract_One year": [1 if contract == "One year" else 0],
            "Contract_Two year": [1 if contract == "Two year" else 0],
            "InternetService_DSL": [1 if internet == "DSL" else 0],
            "InternetService_Fiber optic": [1 if internet == "Fiber optic" else 0],
            "InternetService_No": [1 if internet == "No" else 0]
        }

        for col in X.columns:
            if col not in input_data:
                input_data[col] = [0]

        input_df = pd.DataFrame(input_data)[X.columns]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("📈 Hasil Prediksi")
        if prediction == 1 or probability > 0.45:
            st.error(f"❌ Pelanggan diprediksi akan berhenti berlangganan (churn) dengan probabilitas {probability * 100:.2f}%")
        else:
            st.success(f"✅ Pelanggan diprediksi akan tetap berlangganan (tidak churn) dengan probabilitas {(1 - probability) * 100:.2f}%")

        st.markdown("---")
        st.subheader("📊 Analisis Fitur Input")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Tenure**: {tenure} bulan → {'Rendah' if tenure < 12 else 'Cukup Lama'}")
            st.write(f"**Total Charges**: ${total:.2f} → {'Rendah' if total < 1000 else 'Tinggi'}")
            st.write(f"**Internet Service**: {internet} → {'Berisiko' if internet == 'Fiber optic' else 'Aman'}")
        with col2:
            st.write(f"**Monthly Charges**: ${monthly:.2f} → {'Tinggi' if monthly > 80 else 'Normal'}")
            st.write(f"**Jenis Kontrak**: {contract} → {'Rentan Churn' if contract == 'Month-to-month' else 'Cenderung Aman'}")

        st.markdown("---")
        st.subheader("📌 Kesimpulan")
        kesimpulan = (
            f"Berdasarkan input, pelanggan dengan kontrak **{contract}**, tagihan bulanan **${monthly:.2f}**, "
            f"dan total tagihan **${total:.2f}** diprediksi memiliki kecenderungan **{'CHURN' if prediction == 1 or probability > 0.45 else 'TETAP'}**."
        )
        st.markdown(kesimpulan) 
