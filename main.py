import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from io import BytesIO

st.set_page_config(page_title="Detección de Personalidades", layout="wide")
st.title("🧠 Análisis y Detección de Personalidades")

uploaded_file = st.file_uploader("Cargar archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(10))

    st.success(f"Total de registros cargados: {len(df)}")

    # Filtrar por género
    genero_col = next((col for col in df.columns if "género" in col.lower()), None)
    if genero_col:
        genero_opcion = st.radio("Filtrar por género", ["Ambos", "Masculino", "Femenino"])
        if genero_opcion != "Ambos":
            df = df[df[genero_col].str.lower() == genero_opcion.lower()]

    # Detectar preguntas
    columnas_ignoradas = ["Nombre", "Edad:", "Género:"]
    preguntas = df.select_dtypes(include=["number"]).copy()
    preguntas = preguntas[[col for col in preguntas.columns if col not in columnas_ignoradas]]

    if preguntas.empty:
        st.error("No se encontraron columnas numéricas válidas.")
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

    # Añadir Categoría General según el tipo de personalidad
    categoria_map = {
        "INTJ": "Analistas", "INTP": "Analistas", "ENTJ": "Analistas", "ENTP": "Analistas",
        "INFJ": "Diplomáticos", "INFP": "Diplomáticos", "ENFJ": "Diplomáticos", "ENFP": "Diplomáticos",
        "ISTJ": "Centinelas", "ISFJ": "Centinelas", "ESTJ": "Centinelas", "ESFJ": "Centinelas",
        "ISTP": "Exploradores", "ISFP": "Exploradores", "ESTP": "Exploradores", "ESFP": "Exploradores"
    }

    df["Categoría"] = df["Personalidad"].map(categoria_map)

    st.subheader("Distribución por Categoría de Personalidad (Sin K-Means)")

    categorias_detectadas = df["Categoría"].value_counts()
    seleccionadas_cat = st.multiselect("Selecciona categorías para mostrar", categorias_detectadas.index.tolist(), default=categorias_detectadas.index.tolist())
    df_cat_filtrado = df[df["Categoría"].isin(seleccionadas_cat)].copy()

    fig_cat, ax_cat = plt.subplots()
    sns.countplot(data=df_cat_filtrado, x="Categoría", hue="Personalidad", palette="Set3", ax=ax_cat)
    ax_cat.set_title("Distribución de Personalidades por Categoría (Sin K-Means)")
    ax_cat.set_ylabel("Cantidad")
    ax_cat.legend(title="Personalidad", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_cat)

    fig_cat_buffer = BytesIO()
    fig_cat.savefig(fig_cat_buffer, format="png")
    st.download_button("Descargar gráfico por Categoría", fig_cat_buffer.getvalue(), file_name="grafico_categoria_sin_kmeans.png")

    st.subheader("Conteo de Resultados por Categoría")
    conteo_categoria = df_cat_filtrado["Categoría"].value_counts().reset_index()
    conteo_categoria.columns = ["Categoría", "Cantidad"]
    st.dataframe(conteo_categoria)

    # Análisis sin K-Means
    st.subheader("Distribución de Personalidades (Sin K-Means)")
    tipos_detectados = df_cat_filtrado["Personalidad"].value_counts()
    seleccionados = st.multiselect("Selecciona tipos para mostrar", tipos_detectados.index.tolist(), default=tipos_detectados.index.tolist())
    df_filtrado = df_cat_filtrado[df_cat_filtrado["Personalidad"].isin(seleccionados)]

    fig1, ax1 = plt.subplots()
    sns.barplot(x=df_filtrado["Personalidad"].value_counts().index,
                y=df_filtrado["Personalidad"].value_counts().values,
                palette="Set2", ax=ax1)
    ax1.set_title("Distribución de Personalidades (Sin K-Means)")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

    fig1_buffer = BytesIO()
    fig1.savefig(fig1_buffer, format="png")
    st.download_button("⬇ Descargar gráfico sin K-Means", fig1_buffer.getvalue(), file_name="grafico_sin_kmeans.png")

    # --------------------------------
    # ENTRENAMIENTO K-MEANS sobre categorías seleccionadas
    st.subheader("Clustering con K-Means (No Supervisado)")

    n_clusters = st.slider("Selecciona número de clústeres", min_value=2, max_value=10, value=4)
    preguntas_filtradas = preguntas.loc[df_cat_filtrado.index]
    scaler = StandardScaler()
    preguntas_scaled = scaler.fit_transform(preguntas_filtradas)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cat_filtrado["Cluster"] = kmeans.fit_predict(preguntas_scaled)

    st.success(f"K-Means entrenado con {n_clusters} clústeres.")
    st.write("Distribución por Clúster:")
    st.bar_chart(df_cat_filtrado["Cluster"].value_counts().sort_index())

    st.write("Comparación entre Personalidad y Clúster asignado")
    st.dataframe(pd.crosstab(df_cat_filtrado["Personalidad"], df_cat_filtrado["Cluster"]))

    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_cat_filtrado, x="Cluster", hue="Personalidad", palette="tab10", ax=ax2)
    ax2.set_title("Distribución de Personalidades por Clúster (K-Means)")
    ax2.set_ylabel("Cantidad")
    ax2.legend(title="Personalidad", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)

    fig2_buffer = BytesIO()
    fig2.savefig(fig2_buffer, format="png")
    st.download_button("Descargar gráfico con K-Means", fig2_buffer.getvalue(), file_name="grafico_kmeans.png")

    # PCA 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(preguntas_scaled)
    df_cat_filtrado["PCA1"] = coords[:, 0]
    df_cat_filtrado["PCA2"] = coords[:, 1]

   # Proyectar los centroides al espacio PCA
    centroides_pca = pca.transform(kmeans.cluster_centers_)

    # Crear el gráfico PCA con centroides
    fig3, ax3 = plt.subplots()

    # Gráfico de los puntos por clúster
    sns.scatterplot(data=df_cat_filtrado, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax3)

    # Dibujar los centroides como puntos (círculo) de diferente color (ej. rojo)
    ax3.scatter(centroides_pca[:, 0], centroides_pca[:, 1], 
                s=50, c='red', marker='o', label='Centroide')

    ax3.set_title("Visualización PCA de Clústeres con Centroides")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3)


    # Descargar modelo
    model_buffer = BytesIO()
    pickle.dump(kmeans, model_buffer)
    st.download_button("Descargar modelo K-Means entrenado", model_buffer.getvalue(), file_name="modelo_kmeans.pkl")

    # --------------------------------
    # Descarga de Excel
    st.subheader("Descarga de resultados")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        # Hoja principal con datos
        df_cat_filtrado.to_excel(writer, sheet_name="Datos con Personalidad", index=False)

        # Conteo MBTI
        df_cat_filtrado["Personalidad"].value_counts().to_excel(writer, sheet_name="Conteo MBTI")

        # Conteo por Categoría (modelo MBTI)
        df_cat_filtrado["Categoría"].value_counts().rename_axis("Categoría").reset_index(name="Cantidad")\
            .to_excel(writer, sheet_name="Conteo por Categoría MBTI", index=False)

        # Conteo por Categoría y Clúster (K-Means)
        pd.crosstab(df_cat_filtrado["Categoría"], df_cat_filtrado["Cluster"]).to_excel(writer, sheet_name="Categoría vs Clusters")

        # Matriz MBTI vs Clúster
        pd.crosstab(df_cat_filtrado["Personalidad"], df_cat_filtrado["Cluster"]).to_excel(writer, sheet_name="MBTI vs Clusters")


    st.download_button("Descargar Excel completo", data=excel_buffer.getvalue(), file_name="resultado_personalidades.xlsx")

    st.info("""
    El archivo incluye:
    * Datos con personalidad y clúster (solo categorías seleccionadas)
    * Conteo de tipos MBTI
    * Comparación entre MBTI y K-Means
    También puedes descargar los gráficos y el modelo K-Means entrenado.
    """)
