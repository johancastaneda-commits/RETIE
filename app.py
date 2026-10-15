from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Consulta RETIE",
    page_icon="📘",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = APP_DIR
DOCS_DIR = Path(os.getenv("RETIE_DOCS_DIR", str(DEFAULT_DOCS_DIR)))

# Tamaño aproximado de cada fragmento y superposición entre fragmentos.
# La superposición ayuda a conservar el contexto cuando una idea queda
# repartida entre dos segmentos consecutivos.
CHUNK_SIZE = 2200
CHUNK_OVERLAP = 300

# Número inicial de fragmentos que se recuperan para una pregunta.
TOP_K_DEFAULT = 8

# Límite de caracteres enviados como contexto al modelo.
MAX_CONTEXT_CHARS = 28000

# Modelo configurable desde los Secrets de Streamlit o variables de entorno.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# FUNCIONES DE CONFIGURACIÓN
# ============================================================

def get_setting(name: str, default: str = "") -> str:
    """Lee primero Streamlit Secrets y, si no existe, una variable de entorno."""
    try:
        value = st.secrets.get(name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        # En ejecución local puede no existir el archivo secrets.toml.
        pass

    value = os.getenv(name, default)
    return str(value).strip() if value is not None else default


def get_api_key() -> str:
    return get_setting("GEMINI_API_KEY", "")


def get_model_name() -> str:
    return get_setting("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def get_pdf_files() -> list[Path]:
    """Busca archivos PDF en la carpeta configurada, sin recorrer subcarpetas."""
    if not DOCS_DIR.exists():
        return []
    return sorted(
        [path for path in DOCS_DIR.glob("*.pdf") if path.is_file()],
        key=lambda path: path.name.lower(),
    )


def build_manifest(pdf_files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """Crea una huella para detectar cambios en los PDF."""
    manifest = []
    for path in pdf_files:
        try:
            stat = path.stat()
            manifest.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    return tuple(manifest)


# ============================================================
# EXTRACCIÓN Y FRAGMENTACIÓN DE DOCUMENTOS
# ============================================================

def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en fragmentos con una superposición controlada."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        return [text]

    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks


@st.cache_data(show_spinner=False)
def load_document_index(
    manifest: tuple[tuple[str, int, int], ...]
) -> list[dict[str, Any]]:
    """
    Extrae el texto de cada página y lo divide en fragmentos.
    El manifiesto cambia cuando se modifica un PDF, lo que invalida la caché.
    """
    records: list[dict[str, Any]] = []

    for file_path, _file_size, _modified in manifest:
        path = Path(file_path)
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    result = reader.decrypt("")
                    if result == 0:
                        continue
                except Exception:
                    continue

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                page_text = re.sub(r"\s+", " ", page_text).strip()
                if not page_text:
                    continue

                for chunk_number, chunk in enumerate(split_text(page_text), start=1):
                    records.append(
                        {
                            "document": path.name,
                            "page": page_number,
                            "chunk_number": chunk_number,
                            "text": chunk,
                        }
                    )
        except Exception as exc:
            records.append(
                {
                    "document": path.name,
                    "page": 0,
                    "chunk_number": 0,
                    "text": "",
                    "error": str(exc),
                }
            )

    return records


# ============================================================
# BÚSQUEDA DE FRAGMENTOS RELEVANTES
# ============================================================

def retrieve_relevant_chunks(
    question: str,
    records: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Busca los fragmentos con mayor similitud textual respecto a la pregunta."""
    usable_records = [
        record for record in records
        if record.get("text") and not record.get("error")
    ]

    if not usable_records:
        return []

    corpus = [record["text"] for record in usable_records]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=100000,
        )
        matrix = vectorizer.fit_transform(corpus + [question])
        document_matrix = matrix[:-1]
        question_vector = matrix[-1]
        scores = cosine_similarity(question_vector, document_matrix).flatten()
    except ValueError:
        return []

    ranked_indices = scores.argsort()[::-1]
    results: list[dict[str, Any]] = []

    # Umbral mínimo para evitar enviar fragmentos que no tengan relación
    # textual apreciable con la pregunta.
    minimum_score = 0.015

    for index in ranked_indices:
        score = float(scores[index])
        if score < minimum_score:
            continue

        item = dict(usable_records[int(index)])
        item["score"] = score
        results.append(item)

        if len(results) >= top_k:
            break

    return results


def make_context(chunks: list[dict[str, Any]]) -> str:
    """Construye el contexto para Gemini, respetando un máximo de caracteres."""
    parts: list[str] = []
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        source_label = (
            f"[F{index}] Documento: {chunk['document']} | "
            f"Página: {chunk['page']} | "
            f"Fragmento: {chunk['chunk_number']}"
        )
        part = f"{source_label}\n{chunk['text']}\n"
        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(part) > remaining:
            part = part[:remaining]
            parts.append(part)
            break

        parts.append(part)
        total_chars += len(part)

    return "\n---\n".join(parts)


# ============================================================
# GENERACIÓN DE RESPUESTAS
# ============================================================

def ask_gemini(
    question: str,
    context: str,
    api_key: str,
    model_name: str,
    max_attempts: int = 4,
) -> str:
    """Consulta Gemini y reintenta errores temporales con una espera progresiva."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
Eres un asistente técnico especializado en consulta documental del RETIE.

REGLAS OBLIGATORIAS:
1. Responde en español, de forma clara y técnica.
2. Basa la respuesta únicamente en el CONTEXTO DOCUMENTAL suministrado.
3. No inventes artículos, numerales, tablas, páginas, valores ni requisitos.
4. Incluye referencias en el formato [F1], [F2], etc., junto a las afirmaciones
   que respalden. Usa solo identificadores presentes en el contexto.
5. Cuando sea posible, indica el nombre del documento y la página citada.
6. Si el contexto no contiene información suficiente, dilo expresamente.
   Puedes indicar qué información falta, pero no la completes con suposiciones.
7. Si las fuentes contienen información contradictoria, señala la diferencia
   y cita las fuentes correspondientes; no resuelvas la contradicción por tu cuenta.
8. Diferencia claramente una cita o requisito del documento de cualquier
   explicación interpretativa que hagas a partir del texto recuperado.

CONTEXTO DOCUMENTAL:
{context}

PREGUNTA DEL USUARIO:
{question}

Redacta una respuesta útil y sustentada en las fuentes.
"""

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            answer = getattr(response, "text", None)
            if answer and answer.strip():
                return answer.strip()
            return (
                "El modelo no devolvió texto para esta consulta. "
                "Intenta reformular la pregunta o ampliar el contexto."
            )
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            temporary_error = any(
                marker in message
                for marker in (
                    "503", "429", "500", "502", "504",
                    "unavailable", "resource_exhausted",
                    "deadline", "timeout", "timed out",
                )
            )

            if not temporary_error or attempt == max_attempts - 1:
                break

            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"No fue posible obtener una respuesta de Gemini. "
        f"Verifica la API key, el modelo configurado y la conexión. "
        f"Detalle técnico: {last_error}"
    )


# ============================================================
# INTERFAZ DE STREAMLIT
# ============================================================

st.title("📘 Plataforma de consulta RETIE")
st.write(
    "Consulta el contenido de los documentos RETIE disponibles en la aplicación. "
    "Las respuestas muestran las fuentes documentales recuperadas para facilitar "
    "la verificación."
)

with st.sidebar:
    st.header("Configuración de consulta")
    top_k = st.slider(
        "Número máximo de fragmentos de contexto",
        min_value=3,
        max_value=15,
        value=TOP_K_DEFAULT,
        help=(
            "Define cuántos fragmentos de los documentos RETIE se seleccionarán "
            "como contexto para generar la respuesta."
        ),
    )

    with st.expander("ℹ️ ¿Qué son los fragmentos de contexto?"):
        st.markdown(
            """
            Los **fragmentos de contexto** son porciones de texto extraídas de
            los documentos PDF del RETIE. La aplicación divide el contenido en
            segmentos y busca aquellos que tienen mayor relación con la pregunta.

            **¿Cómo funciona?**

            1. **División documental:** el texto de los PDF se divide en fragmentos
               de aproximadamente **2.200 caracteres**.
            2. **Búsqueda por relevancia:** se comparan las palabras y combinaciones
               de palabras de la pregunta con el contenido de los fragmentos.
            3. **Selección de contexto:** se seleccionan los fragmentos más relevantes
               para enviarlos a Gemini.
            4. **Generación de respuesta:** Gemini analiza el contexto seleccionado
               y redacta una respuesta sustentada en él.

            **¿Qué significa el número seleccionado?**

            Este control establece el número máximo de fragmentos que se intentarán
            utilizar para responder la consulta.

            - **3 a 5 fragmentos:** contexto más reducido; puede servir para preguntas puntuales.
            - **6 a 10 fragmentos:** equilibrio entre contexto y cantidad de información.
            - **11 a 15 fragmentos:** más contexto; puede ayudar en consultas complejas,
              aunque también puede incorporar información menos relevante.

            La cantidad de fragmentos no garantiza la exactitud de la respuesta.
            Si la información pertinente no se recupera de los documentos, la
            respuesta puede ser incompleta.

            **Importante:** cada fragmento conserva el nombre del PDF y el número
            de página de origen para facilitar la verificación de las referencias.
            """
        )

    st.divider()
    st.caption(f"Carpeta de documentos: `{DOCS_DIR}`")
    if st.button("🔄 Actualizar índice de documentos", use_container_width=True):
        load_document_index.clear()
        st.rerun()


api_key = get_api_key()
model_name = get_model_name()
pdf_files = get_pdf_files()
manifest = build_manifest(pdf_files)

if not api_key:
    st.error(
        "No se encontró la clave de Gemini. Configura `GEMINI_API_KEY` "
        "en los Secrets de Streamlit o como variable de entorno."
    )
    st.code('GEMINI_API_KEY = "TU_CLAVE_DE_API"', language="toml")
    st.stop()

if not pdf_files:
    st.warning(
        f"No se encontraron archivos PDF en `{DOCS_DIR}`. "
        "Agrega allí los documentos RETIE y vuelve a cargar la aplicación."
    )
    st.stop()

with st.spinner("Leyendo documentos y preparando el índice de consulta..."):
    records = load_document_index(manifest)

valid_records = [
    record for record in records
    if record.get("text") and not record.get("error")
]
failed_records = [
    record for record in records
    if record.get("error")
]

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("PDF encontrados", len(pdf_files))
metric_2.metric("Fragmentos indexados", len(valid_records))
metric_3.metric("Modelo configurado", model_name)

if failed_records:
    with st.expander("Ver advertencias de lectura de PDF"):
        for record in failed_records:
            st.warning(
                f"No se pudo procesar **{record['document']}**: "
                f"{record.get('error', 'error desconocido')}"
            )

if not valid_records:
    st.error(
        "No se pudo extraer texto de los PDF. Comprueba que los documentos "
        "contengan texto seleccionable y no sean únicamente imágenes escaneadas."
    )
    st.stop()

st.divider()
st.subheader("Realizar una consulta")

with st.form("retie_question_form"):
    question = st.text_area(
        "Escribe tu pregunta sobre el RETIE",
        placeholder=(
            "Ejemplo: ¿Qué requisitos se establecen para la puesta a tierra "
            "de una instalación eléctrica?"
        ),
        height=110,
    )
    submitted = st.form_submit_button("Consultar documentos", type="primary")

if submitted:
    if not question.strip():
        st.warning("Escribe una pregunta antes de realizar la consulta.")
    else:
        with st.spinner("Buscando fragmentos relevantes en los documentos..."):
            relevant_chunks = retrieve_relevant_chunks(
                question=question.strip(),
                records=valid_records,
                top_k=top_k,
            )

        if not relevant_chunks:
            st.warning(
                "No se encontraron fragmentos con suficiente relación textual "
                "con la pregunta. Prueba con otros términos o formula la pregunta "
                "de manera más específica."
            )
        else:
            context = make_context(relevant_chunks)

            try:
                with st.spinner("Analizando las fuentes y generando la respuesta..."):
                    answer = ask_gemini(
                        question=question.strip(),
                        context=context,
                        api_key=api_key,
                        model_name=model_name,
                    )

                st.session_state["last_question"] = question.strip()
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = relevant_chunks

            except Exception as exc:
                st.error(str(exc))

if st.session_state.get("last_answer"):
    st.divider()
    st.subheader("Respuesta")
    st.markdown(st.session_state["last_answer"])

    st.subheader("Fragmentos documentales consultados")
    st.caption(
        "Estos son los fragmentos recuperados para esta consulta. "
        "Verifica el texto en el PDF original antes de tomar decisiones técnicas "
        "o normativas."
    )

    for index, source in enumerate(st.session_state.get("last_sources", []), start=1):
        title = (
            f"[F{index}] {source['document']} — página {source['page']} "
            f"— relevancia {source.get('score', 0):.3f}"
        )
        with st.expander(title):
            st.write(source["text"])

    if st.button("Limpiar respuesta"):
        for key in ("last_question", "last_answer", "last_sources"):
            st.session_state.pop(key, None)
        st.rerun()

st.divider()
st.caption(
    "Herramienta de consulta documental. Las respuestas generadas por IA pueden "
    "ser incompletas; confirma siempre los requisitos en el documento RETIE original."
)
