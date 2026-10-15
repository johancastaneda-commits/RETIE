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
