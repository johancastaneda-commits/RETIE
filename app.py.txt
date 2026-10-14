import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="Asistente de Consulta Documental", page_icon="⚡", layout="wide"
)

st.title("⚡ Plataforma de Consulta Interactiva de Documentos")
st.markdown(
    "Sube tus archivos PDF, realiza consultas por texto y obtén respuestas"
    " precisas con la referencia exacta del documento y página."
)

# BARRA LATERAL: CARGA DE ARCHIVOS PDF
st.sidebar.header("📁 Carga de Documentos")
uploaded_files = st.sidebar.file_uploader(
    "Sube tus 4 archivos PDF aquí",
    type=["pdf"],
    accept_multiple_files=True,
)

# Indexar documentos cargados en memoria
document_database = []

if uploaded_files:
  for uploaded_file in uploaded_files:
    reader = PdfReader(uploaded_file)
    for page_num, page in enumerate(reader.pages):
      text = page.extract_text()
      if text:
        document_database.append({
            "doc_name": uploaded_file.name,
            "page": page_num + 1,
            "content": text,
        })
  st.sidebar.success(
      f"¡{len(uploaded_files)} documentos cargados y analizados con éxito!"
      f" ({len(document_database)} páginas indexadas)."
  )
else:
  st.sidebar.info("👈 Por favor, carga tus archivos PDF para comenzar.")

# ZONA PRINCIPAL DE CONSULTA
if document_database:
  st.markdown("---")
  query = st.text_input(
      "🔍 Escribe tu duda o pregunta sobre la documentación:",
      placeholder=(
          "Ej: ¿Cuáles son las distancias de seguridad o parámetros de"
          " configuración?"
      ),
  )

  if query:
    query_lower = query.lower()
    resultados = []

    # Buscar coincidencias en las páginas de los PDFs
    for item in document_database:
      if query_lower in item["content"].lower():
        resultados.append(item)

    if resultados:
      st.success(
          f"Se encontraron {len(resultados)} coincidencias en tus documentos:"
      )
      for i, res in enumerate(resultados[:6]):  # Mostrar máximo 6 resultados
        with st.expander(
            f"📄 Coincidencia #{i+1} en '{res['doc_name']}' - Página"
            f" {res['page']}",
            expanded=True,
        ):
          # Extraer fragmento relevante
          idx = res["content"].lower().find(query_lower)
          inicio = max(0, idx - 100)
          fin = min(len(res["content"]), idx + 350)
          fragmento = res["content"][inicio:fin].replace("\n", " ")

          st.markdown(
              f"**Texto de referencia:** *...{fragmento}...*"
          )
          st.markdown(
              f"📌 **Referencia exacta:** Documento: **`{res['doc_name']}`** |"
              f" Página: **{res['page']}**"
          )
    else:
      st.warning(
          "No se encontraron coincidencias exactas para ese término en los"
          " documentos cargados. Prueba con otras palabras clave."
      )
else:
  st.info(
      "La aplicación está lista. Sube tus PDFs en la barra lateral para"
      " habilitar el buscador de consultas."
  )