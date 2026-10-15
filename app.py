import os
import time
from google import genai
import pypdf
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Asistente Normativo RETIE Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo visual moderno con CSS personalizado
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stChatInputContainer { padding-bottom: 1rem; }
    .source-box { background-color: #f1f5f9; border-left: 4px solid #f59e0b; padding: 10px; border-radius: 4px; font-size: 0.85rem; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Asistente Inteligente RETIE Pro")
st.markdown(
    "Plataforma avanzada de consulta normativa asistida por Inteligencia"
    " Artificial con referencias cruzadas exactas."
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


# 2. FUNCIÓN DE PROCESAMIENTO AVANZADO (CHUNKING Y CACHÉ)
@st.cache_resource
def inicializar_base_conocimiento():
  database = []
  carpeta_docs = "."
  archivos = [
      f for f in os.listdir(carpeta_docs) if f.lower().endswith(".pdf")
  ]

  if not archivos:
    return database, 0, 0, "No se encontraron archivos PDF en el repositorio."

  total_paginas = 0
  chunk_size = 800  # Caracteres por fragmento
  overlap = 150  # Solapamiento para no perder contexto entre fragmentos

  for archivo in archivos:
    ruta_archivo = os.path.join(carpeta_docs, archivo)
    try:
      reader = pypdf.PdfReader(ruta_archivo)
      total_paginas += len(reader.pages)
      for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
          # Dividir la página en fragmentos (chunks) más precisos
          for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk.strip()) > 40:
              database.append({
                  "doc_name": archivo,
                  "page": page_num + 1,
                  "content": chunk,
              })
    except Exception as e:
      st.error(f"Error al procesar {archivo}: {e}")

  mensaje = (
      f"¡{len(archivos)} documentos procesados! ({total_paginas} páginas,"
      f" {len(database)} fragmentos indexados)."
  )
  return database, len(archivos), total_paginas, mensaje


# CARGAR BASE DOCUMENTAL
with st.spinner("🔄 Indexando normativa y preparando base de conocimiento..."):
  document_database, num_docs, num_paginas, mensaje_estado = (
      inicializar_base_conocimiento()
  )

# --- BARRA LATERAL ---
st.sidebar.header("📊 Estado del Sistema")
if document_database:
  st.sidebar.success(mensaje_estado)
  st.sidebar.metric("Documentos Activos", num_docs)
  st.sidebar.metric("Páginas Totales", num_paginas)
else:
  st.sidebar.warning(mensaje_estado)

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

# Botón para limpiar historial de chat
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Limpiar Conversación"):
  st.session_state.messages = []
  st.rerun()

# --- ZONA PRINCIPAL DE CHAT CON IA ---
if not document_database:
  st.warning(
      "⚠️ Por favor, asegúrate de tener los 4 archivos PDF en la raíz del"
      " repositorio."
  )
else:
  client = genai.Client(api_key=api_key)

  # Inicializar historial de chat en sesión
  if "messages" not in st.session_state:
    st.session_state.messages = []

  # Mostrar mensajes anteriores del chat
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      if "sources" in message and message["sources"]:
        with st.expander("📌 Ver referencias y fragmentos normativos utilizados"):
          for src in message["sources"]:
            st.markdown(
                f"<div class='source-box'><b>Documento:</b> {src['doc_name']} |"
                f" <b>Página:</b> {src['page']}<br><em>...{src['snippet']}...</em></div>",
                unsafe_allow_html=True,
            )

  # Sugerencias rápidas de consulta
  st.markdown("##### 💡 Consultas Sugeridas:")
  col1, col2, col3 = st.columns(3)
  query_sugerida = None

  with col1:
    if st.button("🔌 Requisitos de conductores", use_container_width=True):
      query_sugerida = "¿Cuáles son los requisitos de certificación para los conductores eléctricos?"
  with col2:
    if st.button("📏 Distancias de seguridad", use_container_width=True):
      query_sugerida = "¿Cuáles son las distancias mínimas de seguridad establecidas?"
  with col3:
    if st.button("🏢 Obligaciones de inspectores", use_container_width=True):
      query_sugerida = "¿Quiénes están facultados para realizar inspecciones de conformidad?"

  # Capturar entrada del usuario (vía chat input o botón sugerido)
  user_query = st.chat_input(
      "Escribe tu duda técnica sobre el RETIE y la normativa..."
  )

  if query_sugerida:
    user_query = query_sugerida

  if user_query:
    # Registrar mensaje del usuario
    st.session_state.messages.append(
        {"role": "user", "content": user_query, "sources": []}
    )
    with st.chat_message("user"):
      st.markdown(user_query)

    # Generar respuesta de la IA
    with st.chat_message("assistant"):
      with st.spinner("🤖 Analizando normativa y redactando informe..."):
        try:
          # 1. Búsqueda semántica / por coincidencia de palabras clave en los fragmentos
          query_words = user_query.lower().split()
          scored_chunks = []

          for item in document_database:
            score = sum(
                1 for word in query_words if word in item["content"].lower()
            )
            if score > 0:
              scored_chunks.append((score, item))

          # Si no hay coincidencia exacta estricta, tomamos los primeros fragmentos para asegurar contexto
          if not scored_chunks:
            scored_chunks = [(1, item) for item in document_database[:10]]

          # Ordenar por relevancia y seleccionar los mejores 6 fragmentos únicos
          scored_chunks.sort(key=lambda x: x[0], reverse=True)
          top_chunks = []
          seen_pages = set()

          for score, item in scored_chunks:
            identifier = (item["doc_name"], item["page"])
            if identifier not in seen_pages:
              seen_pages.add(identifier)
              top_chunks.append(item)
            if len(top_chunks) >= 6:
              break

          # Construir contexto para la IA
          context_text = ""
          sources_metadata = []
          for p in top_chunks:
            context_text += f"\n--- [Documento: {p['doc_name']} | Página: {p['page']}] ---\n{p['content']}\n"
            sources_metadata.append({
                "doc_name": p["doc_name"],
                "page": p["page"],
                "snippet": p["content"][:200].replace("\n", " "),
            })

          # Prompt profesional especializado
          prompt = f"""
                        Eres un ingeniero senior experto consultor en telecomunicaciones, electricidad y regulaciones del RETIE (Reglamento Técnico de Instalaciones Eléctricas).
                        Analiza detalladamente el contexto extraído de los libros normativos oficiales y responde de manera técnica, estructurada y precisa a la duda del usuario.
                        
                        REGLAS ESTRICTAS:
                        1. Basa tu respuesta exclusivamente en el contexto provisto de los documentos PDF.
                        2. Explica con claridad los conceptos técnicos involucrados.
                        3. Al finalizar tu respuesta, añade obligatoriamente una sección clara de **Referencias Bibliográficas** listando los documentos y páginas específicos utilizados.
                        
                        Pregunta técnica: {user_query}

                        Contexto normativo extraído:
                        {context_text}
                        """

          # 2. Llamada a la API con reintento automático ante saturación (503)
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

          ai_response_text = response.text
          st.markdown(ai_response_text)

          # Mostrar referencias expandibles debajo del mensaje del asistente
          if sources_metadata:
            with st.expander(
                "📌 Ver referencias y fragmentos normativos utilizados"
            ):
              for src in sources_metadata:
                st.markdown(
                    f"<div class='source-box'><b>Documento:</b>"
                    f" {src['doc_name']} | <b>Página:</b>"
                    f" {src['page']}<br><em>...{src['snippet']}...</em></div>",
                    unsafe_allow_html=True,
                )

          # Guardar en el historial de sesión
          st.session_state.messages.append({
              "role": "assistant",
              "content": ai_response_text,
              "sources": sources_metadata,
          })

        except Exception as e:
          error_msg = f"Ocurrió un error al procesar la consulta con la inteligencia artificial: {e}"
          st.error(error_msg)
