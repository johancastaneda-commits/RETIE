from google import genai
import pypdf
import streamlit as st

st.set_page_config(
    page_title="Asistente Normativo con Gemini AI", page_icon="⚡", layout="wide"
)

st.title("⚡ Asistente de Consulta RETIE con Inteligencia Artificial")
st.markdown(
    "Sube tus documentos PDF, ingresa tu clave de la API de Google y realiza"
    " cualquier consulta técnica en lenguaje natural."
)

# BARRA LATERAL: CONFIGURACIÓN Y ARCHIVOS
st.sidebar.header("⚙️ Configuración")

# 1. Input para la API Key de Google
api_key = st.sidebar.text_input(
    "🔑 Google Gemini API Key",
    type="password",
    help=(
        "Obtén tu clave gratis en Google AI Studio (aistudio.google.com)"
    ),
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Carga de Documentos")
uploaded_files = st.sidebar.file_uploader(
    "Sube tus archivos PDF aquí", type=["pdf"], accept_multiple_files=True
)

# Indexar documentos en memoria
document_database = []
if uploaded_files:
  for uploaded_file in uploaded_files:
    reader = pypdf.PdfReader(uploaded_file)
    for page_num, page in enumerate(reader.pages):
      text = page.extract_text()
      if text:
        document_database.append({
            "doc_name": uploaded_file.name,
            "page": page_num + 1,
            "content": text,
        })
  st.sidebar.success(
      f"¡{len(uploaded_files)} documentos indexados ({len(document_database)}"
      f" páginas)!"
  )
else:
  st.sidebar.info("👈 Sube tus PDFs en la barra lateral.")

# ZONA PRINCIPAL DE CONSULTA CON IA
if not api_key:
  st.warning(
      "⚠️ Por favor, ingresa tu **Google Gemini API Key** en la barra lateral"
      " para activar el asistente de IA."
  )
elif not document_database:
  st.info(
      "La aplicación está lista. Sube tus documentos PDF en la barra lateral"
      " para comenzar a chatear con la IA."
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

        # Construcción del prompt estructurado para la IA
        prompt = f"""
                Eres un ingeniero electricista senior y experto consultor en el RETIE (Reglamento Técnico de Instalaciones Eléctricas).
                Analiza el siguiente contexto extraído de los documentos del usuario y responde de forma profesional, clara y detallada a la pregunta.
                
                REGLAS OBLIGATORIAS:
                1. Basate estrictamente en el contexto provisto de los documentos PDF.
                2. Al finalizar tu respuesta, incluye un apartado de **Referencias Exactas** indicando claramente el nombre del documento y el número de página de donde obtuviste la información.
                
                Pregunta del usuario: {user_query}

                Contexto de los documentos:
                {context_text}
                """

        try:
          # Llamada al modelo Gemini de Google
          response = client.models.generate_content(
              model="gemini-2.5-flash",
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