import os
import time
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

# 1. CARGAR LA API KEY DE FORMA SEGURA DESDE LOS SECRETOS DE STREAMLIT
try:
  api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error(
      "⚠️ **Error de Configuración:** No se encontró la `GEMINI_API_KEY` en"
      " los secretos de Streamlit. Por favor, configúrala en el panel de"
      " administración de tu app."
  )
  st.stop()


# 2. FUNCIÓN PARA CARGAR Y CACHEAR LOS PDFs DESDE LA RAIZ DEL REPOSITORIO
@st.cache_resource
def cargar_documentos_precargados():
  document_database = []
  carpeta_docs = "."

  archivos = [
      f for f in os.listdir(carpeta_docs) if f.lower().endswith(".pdf")
  ]

  if not archivos:
    return (
        document_database,
        "No se encontraron archivos PDF en el repositorio.",
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
      f"¡{len(archivos)} documentos cargados y listos! ({len(document_database)}"
      f" páginas indexadas).",
  )


# BARRA LATERAL: ESTADO DE LA BASE DOCUMENTAL Y LISTADO OFICIAL
st.sidebar.header("📁 Base Documental")
document_database, mensaje_estado = cargar_documentos_precargados()

if document_database:
  st.sidebar.success(mensaje_estado)
else:
  st.sidebar.warning(mensaje_estado)

# Sección informativa de los libros utilizados en el menú lateral
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Normativa Indexada:")
st.sidebar.markdown("""
- **Libro 1.** Aspectos Generales  
  *(Resolución 40284 de 2026)*
- **Libro 2.** Productos objeto del RETIE  
  *(Resolución 40284 de 2026)*
- **Libro 3.** Instalaciones objeto del RETIE  
  *(Resolución 40284 de 2026)*
- **Libro 4.** Evaluación de la conformidad  
  *(Resolución 40284 de 2026)*
""")

# ZONA PRINCIPAL DE CONSULTA CON IA
if not document_database:
  st.info(
      "La aplicación está esperando a que coloques los archivos PDF en el"
      " repositorio."
  )
else:
  # Inicializar cliente de Google GenAI de forma privada
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
          # SISTEMA DE REINTENTO AUTOMÁTICO ANTE ERRORES 503 DE ALTA DEMANDA
          intentos = 3
          response = None
          for intento in range(intentos):
            try:
              response = client.models.generate_content(
                  model="gemini-3.8-flash",
                  contents=prompt,
              )
              break
            except Exception as api_err:
              if "503" in str(api_err) and intento < intentos - 1:
                time.sleep(3)
                continue
              else:
                raise api_err

          st.markdown("### 📋 Análisis y Respuesta:")
          st.markdown(response.text)

        except Exception as e:
          st.error(
              f"Ocurrió un error al conectar con la API de Google Gemini (los"
              f" servidores están ocupados): {e}"
          )
    else:
      st.warning("Por favor, escribe una pregunta válida.")
