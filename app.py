import os
from google import genai
import pypdf
import streamlit as st

st.set_page_config(
    page_title="Asistente Normativo con Gemini AI", page_icon="⚡", layout="wide"
)

st.title("⚡ Asistente de Consulta RETIE con Inteligencia Artificial")
st.markdown(
    "Plataforma inteligente con normativa precargada para consultas técnicas"
    " en lenguaje natural y referencias bibliográficas exactas."
)


# FUNCIÓN PARA CARGAR Y CACHEAR LOS PDFs DESDE LA CARPETA DEL REPOSITORIO
@st.cache_resource
def cargar_documentos_precargados():
  document_database = []
  carpeta_docs = "documentos"

  if not os.path.exists(carpeta_docs):
    os.makedirs(carpeta_docs)
    return (
        document_database,
        f"La carpeta '{carpeta_docs}' no existe. Sube tus PDFs allí en GitHub.",
    )

  archivos = [f for f in os.listdir(carpeta_docs) if f.lower().endswith(".pdf")]

  if not archivos:
    return (
        document_database,
        f"No se encontraron archivos PDF dentro de la carpeta '{carpeta_docs}'.",
    )

  for archivo in archivos:
    ruta_archivo = os.path.join(carpeta_docs, archivo)
    try:
      reader = pypdf.PdfReader(ruta_archivo)
      for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
          document_database.append({
              "doc_name": archivo,
              "page": page_num + 1,
              "content": text,
          })
    except Exception as e:
      st.error(f"Error al leer el archivo {archivo}: {e}")

  return (
      document_database,
      f"¡{len(archivos)} documentos precargados y listos! ({len(document_database)}"
      f" páginas indexadas).",
  )


# BARRA LATERAL: CONFIGURACIÓN Y ESTADO DE LOS DOCUMENTOS
st.sidebar.header("⚙️ Configuración")
api_key = st.sidebar.text_input(
    "🔑 Google Gemini API Key",
    type="password",
    help="Obtén tu clave gratis en Google AI Studio (aistudio.google.com)",
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Base Documental")

# Cargar documentos de forma automática
document_database, mensaje_estado = cargar_documentos_precargados()

if document_database:
  st.sidebar.success(mensaje_estado)
else:
  st.sidebar.warning(mensaje_estado)
  st.sidebar.info(
      "ℹ️ Asegúrate de crear una carpeta llamada **'documentos'** en tu"
      " repositorio de GitHub y subir allí tus archivos PDF."
  )

# ZONA PRINCIPAL DE CONSULTA CON IA
if not api_key:
  st.warning(
      "⚠️ Por favor, ingresa tu **Google Gemini API Key** en la barra lateral"
      " para activar el asistente de IA."
  )
elif not document_database:
  st.info(
      "La aplicación está esperando a que coloques los archivos PDF en la"
      " carpeta **'documentos'** de tu repositorio."
  )
else:
  # Inicializar cliente de Google GenAI
  client = genai.Client(api_key=api_key)

  st.markdown("---")
  user_query = st.text_area(
      "💬 Escribe tu duda o pregunta sobre la normativa:",
      placeholder=(
          "Ej: ¿Cuáles son los requisitos de certificación para los"
          " conductores y qué excepciones aplican?"
      ),
  )

  if st.button("Consultar con IA 🚀", type="primary"):
    if user_query.strip():
      with st.spinner(
          "🤖 Analizando documentos y redactando respuesta técnica..."
      ):
        # Búsqueda preliminar de páginas relevantes para pasar como contexto a Gemini
        query_words = user_query.lower().split()
        relevance_scores = []

        for item in document_database:
          score = sum(
              1 for word in query_words if word in item["content"].lower()
          )
          relevance_scores.append((score, item))

        # Ordenar y seleccionar las 8 páginas más relevantes para el contexto
        relevance_scores.sort(key=lambda x: x[0], reverse=True)
        top_pages = [item for score, item in relevance_scores[:8]]

        context_text = ""
        for p in top_pages:
          context_text += f"\n--- [Documento: {p['doc_name']} | Página: {p['page']}] ---\n{p['content']}\n"

        # Prompt estructurado para la IA
        prompt = f"""
                Eres un ingeniero electricista senior y experto consultor en el RETIE (Reglamento Técnico de Instalaciones Eléctricas).
                Analiza el siguiente contexto extraído de los documentos normativos y responde de forma profesional, clara y detallada a la pregunta.
                
                REGLAS OBLIGATORIAS:
                1. Basate estrictamente en el contexto provisto de los documentos PDF.
                2. Al finalizar tu respuesta, incluye un apartado de **Referencias Exactas** indicando claramente el nombre del documento y el número de página de donde obtuviste la información.
                
                Pregunta del usuario: {user_query}

                Contexto de los documentos:
                {context_text}
                """

        try:
          # Llamada actualizada al modelo correcto de Google Gemini
          response = client.models.generate_content(
              model="gemini-3.8-flash",
              contents=prompt,
          )

          st.markdown("### 📋 Análisis y Respuesta:")
          st.markdown(response.text)

        except Exception as e:
          st.error(
              f"Ocurrió un error al conectar con la API de Google Gemini: {e}"
          )
    else:
      st.warning("Por favor, escribe una pregunta válida.")
