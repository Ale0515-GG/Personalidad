import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from io import BytesIO

st.set_page_config(page_title="Detección de Personalidades MBTI", layout="wide")
st.title("🧠 Análisis y Detección de Personalidades MBTI")

uploaded_file = st.file_uploader("📤 Cargar archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Vista previa de los datos")
    st.dataframe(df.head())

    st.success(f"📈 Total de registros cargados: {len(df)}")

    # Filtrar por género
    genero_col = next((col for col in df.columns if "género" in col.lower()), None)
    if genero_col:
        genero_opcion = st.radio("👥 Filtrar por género", ["Ambos", "Masculino", "Femenino"])
        if genero_opcion != "Ambos":
            df = df[df[genero_col].str.lower() == genero_opcion.lower()]

    # Detectar preguntas
    columnas_ignoradas = ["Nombre", "Género", "Edad", "Edad:", "Género:"]
    preguntas = df.select_dtypes(include=["number"]).copy()
    preguntas = preguntas[[col for col in preguntas.columns if col not in columnas_ignoradas]]

    if preguntas.empty:
        st.error("❌ No se encontraron columnas numéricas válidas.")
        st.stop()

    # Mapeo MBTI
    dimension_map = {
        "EI": preguntas.iloc[:, [0, 1, 2, 3, 4]],
        "SN": preguntas.iloc[:, [5, 6, 7, 8, 9]],
        "TF": preguntas.iloc[:, [10, 11, 12, 13, 14]],
        "JP": preguntas.iloc[:, [15, 16, 17, 18, 19]],
    }

    def calcular_mbti(fila):
        letras = ""
        letras += "I" if dimension_map["EI"].loc[fila.name].mean() > 3 else "E"
        letras += "N" if dimension_map["SN"].loc[fila.name].mean() > 3 else "S"
        letras += "F" if dimension_map["TF"].loc[fila.name].mean() > 3 else "T"
        letras += "P" if dimension_map["JP"].loc[fila.name].mean() > 3 else "J"
        return letras

    df["Personalidad"] = preguntas.apply(calcular_mbti, axis=1)

    # Análisis sin K-Means
    st.subheader("📊 Distribución de Personalidades (Sin K-Means)")
    tipos_detectados = df["Personalidad"].value_counts()
    seleccionados = st.multiselect("Selecciona tipos para mostrar", tipos_detectados.index.tolist(), default=tipos_detectados.index.tolist())
    df_filtrado = df[df["Personalidad"].isin(seleccionados)]

    fig1, ax1 = plt.subplots()
    sns.barplot(x=df_filtrado["Personalidad"].value_counts().index, 
                y=df_filtrado["Personalidad"].value_counts().values,
                palette="Set2", ax=ax1)
    ax1.set_title("Distribución de Personalidades (Sin K-Means)")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

    # Descargar gráfico sin K-Means
    fig1_buffer = BytesIO()
    fig1.savefig(fig1_buffer, format="png")
    st.download_button("⬇️ Descargar gráfico sin K-Means", fig1_buffer.getvalue(), file_name="grafico_sin_kmeans.png")

    # --------------------------------
    # ENTRENAMIENTO K-MEANS
    st.subheader("🤖 Clustering con K-Means (No Supervisado)")

    n_clusters = st.slider("Selecciona número de clústeres", min_value=2, max_value=10, value=4)
    scaler = StandardScaler()
    preguntas_scaled = scaler.fit_transform(preguntas)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(preguntas_scaled)

    st.success(f"🎯 K-Means entrenado con {n_clusters} clústeres.")
    st.write("🔢 Distribución por Clúster:")
    st.bar_chart(df["Cluster"].value_counts().sort_index())

    # Comparación entre MBTI y clústeres
    st.write("📌 Comparación entre Personalidad y Clúster asignado")
    st.dataframe(pd.crosstab(df["Personalidad"], df["Cluster"]))

    # Gráfico de barras K-Means
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Cluster", hue="Personalidad", palette="tab10", ax=ax2)
    ax2.set_title("Distribución de Personalidades por Clúster (K-Means)")
    ax2.set_ylabel("Cantidad")
    st.pyplot(fig2)

    fig2_buffer = BytesIO()
    fig2.savefig(fig2_buffer, format="png")
    st.download_button("⬇️ Descargar gráfico con K-Means", fig2_buffer.getvalue(), file_name="grafico_kmeans.png")

    # PCA 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(preguntas_scaled)
    df["PCA1"] = coords[:, 0]
    df["PCA2"] = coords[:, 1]

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax3)
    ax3.set_title("🔍 Visualización PCA de Clústeres")
    st.pyplot(fig3)

    # Descargar modelo
    model_buffer = BytesIO()
    pickle.dump(kmeans, model_buffer)
    st.download_button("⬇️ Descargar modelo K-Means entrenado", model_buffer.getvalue(), file_name="modelo_kmeans.pkl")

    # --------------------------------
    # Descarga de Excel
    st.subheader("📂 Descarga de resultados")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Datos con Personalidad", index=False)
        df["Personalidad"].value_counts().to_excel(writer, sheet_name="Conteo MBTI")
        pd.crosstab(df["Personalidad"], df["Cluster"]).to_excel(writer, sheet_name="MBTI vs Clusters")

    st.download_button("⬇️ Descargar Excel completo", data=excel_buffer.getvalue(), file_name="resultado_personalidades.xlsx")

    st.info("""
    El archivo incluye:
    ✅ Datos con personalidad y clúster
    ✅ Conteo de tipos MBTI
    ✅ Comparación entre MBTI y K-Means
    También puedes descargar los gráficos y el modelo K-Means entrenado.
    """)
