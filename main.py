import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fpdf import FPDF
import os

# Crear carpeta resultados si no existe
os.makedirs("resultados", exist_ok=True)

# 1. Cargar datos
df = pd.read_excel("Respuestas.xlsx")
columnas_preguntas = [col for col in df.columns if col.startswith("Q")]
X = df[columnas_preguntas]

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. KMeans
k = 16
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 4. Asignar personalidad a cada cluster
cluster_personalidad = {
    0: "Arquitecto (INTJ)",
    1: "Defensor (ISFJ)",
    2: "Ejecutivo (ESTJ)",
    3: "Activista (ENFP)",
    4: "Logista (ISTJ)",
    5: "Lógico (INTP)",
    6: "Cónsul (ESFJ)",
    7: "Mediador (INFP)",
    8: "Animador (ESFP)",
    9: "Protagonista (ENFJ)",
    10: "Abogado (INFJ)",
    11: "Virtuoso (ISTP)",
    12: "Emprendedor (ESTP)",
    13: "Comandante (ENTJ)",
    14: "Innovador (ENTP)",
    15: "Aventurero (ISFP)"
}
df["Personalidad"] = df["Cluster"].map(cluster_personalidad)

# 5. Guardar resultados en Excel
df.to_excel("resultados/resultados.xlsx", index=False)
print("Resultados guardados en resultados/resultados.xlsx")

# 6. PCA para graficar
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

# 7. Gráfico de dispersión
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Personalidad", palette="tab20", s=100)
plt.title("Clústeres de Personalidades - KMeans")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("resultados/grafico.png")
plt.show()

# 8. Crear reporte PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Reporte de Agrupación de Personalidades (K-Means)", ln=True, align="C")
        self.ln(10)

    def tabla(self, data):
        self.set_font("Arial", size=9)
        col_width = self.w / (len(data.columns) + 1)
        self.set_fill_color(220, 220, 255)

        # Encabezado
        for col in data.columns:
            self.cell(col_width, 8, str(col), border=1, fill=True)
        self.ln()

        # Filas
        for _, row in data.iterrows():
            for item in row:
                self.cell(col_width, 8, str(item), border=1)
            self.ln()

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, "Distribución visual de los clústeres:", ln=True)
pdf.image("resultados/grafico.png", x=10, w=190)
pdf.ln(10)
pdf.cell(0, 10, "Tabla de resultados (primeros 10 usuarios):", ln=True)
pdf.tabla(df[["Cluster", "Personalidad"] + columnas_preguntas].head(10))
pdf.output("resultados/reporte.pdf")
print("📄 Reporte PDF generado en resultados/reporte.pdf")
