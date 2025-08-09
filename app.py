import os
import signal
from datetime import datetime
import streamlit as st

from rag_chain import rag_chain, build_context, get_sources  # from your rag_chain.py

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="RAG Academic Chat", layout="wide")
st.title("📚 Academic Chat with Wouter's Zotero Library")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

if "answers_path" not in st.session_state:
    os.makedirs("answers", exist_ok=True)
    st.session_state.answers_path = f"answers/answers_{st.session_state.session_started_at}.org"
    # Initialize the org file
    with open(st.session_state.answers_path, "w", encoding="utf-8") as f:
        f.write("#+OPTIONS: toc:nil author:nil\n")
        f.write(f"#+TITLE: Answers and sources for session started on {st.session_state.session_started_at}\n\n")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.markdown("### Session")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # --- Quit app ---
    st.markdown("---")
    if "confirm_quit" not in st.session_state:
        st.session_state.confirm_quit = False

    if not st.session_state.confirm_quit:
        if st.button("🛑 Quit app"):
            st.session_state.confirm_quit = True
            st.rerun()
    else:
        st.warning("Really quit the app for all users?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, quit"):
                os.kill(os.getpid(), signal.SIGTERM)  # clean terminate
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_quit = False
                st.rerun()

    st.markdown("---")
    st.markdown("**Log file:**")
    st.code(st.session_state.answers_path, language="text")
    st.caption("Each Q&A is appended here.")

# ----------------------------
# Render chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {s['ref']}**")
                    st.markdown(f"`{s['filename']}`")
                    st.markdown(f"> {s['content'][:1000]}{'…' if len(s['content'])>1000 else ''}")
                    st.markdown("---")

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask a question about your papers…")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer…"):
            # Build retrieval context and get sources
            context_str, docs = build_context(user_input)
            sources = get_sources(docs)

            # Call the LCEL chain (expects question, context, chat_history)
            # For now we pass empty chat_history; you can inject memory later.
            answer = rag_chain.invoke({
                "question": user_input,
                "context": context_str,
                "chat_history": ""
            })

            # Compose full message with a Sources section beneath
            full_answer = answer
            if sources:
                full_answer += "\n\n### Sources\n"
                for idx, src in enumerate(sources, 1):
                    full_answer += f"**{idx}. {src['ref']}**\n\n"
                    full_answer += f"`{src['filename']}`\n\n"
                    # Keep the preview compact in the log
                    preview = src["content"].replace("\n", " ")
                    if len(preview) > 1200:
                        preview = preview[:1200] + "…"
                    full_answer += f"> {preview}\n\n"

            # Render to UI
            st.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=True):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"**{i}. {s['ref']}**")
                        st.markdown(f"`{s['filename']}`")
                        st.markdown(f"> {s['content'][:1500]}{'…' if len(s['content'])>1500 else ''}")
                        st.markdown("---")
            else:
                st.info("No relevant documents were retrieved for this query.")

            # Append to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Append to .org log
            try:
                with open(st.session_state.answers_path, "a", encoding="utf-8") as f:
                    f.write(f"* Question:\n{user_input}\n\n")
                    f.write(f"* Answer:\n{answer}\n\n")
                    if sources:
                        f.write("* Sources:\n")
                        for idx, src in enumerate(sources, 1):
                            f.write(f"** {idx}. {src['ref']}\n")
                            f.write(f"   - {src['filename']}\n")
                            # Keep logs compact
                            preview = src["content"].replace("\n", " ")
                            if len(preview) > 1500:
                                preview = preview[:1500] + "…"
                            f.write(f"   - {preview}\n")
                        f.write("\n")
            except Exception as e:
                st.warning(f"Could not write to log file: {e}")
