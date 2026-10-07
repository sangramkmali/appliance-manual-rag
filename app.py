"""Streamlit chat UI:  streamlit run app.py"""
import streamlit as st
from rag import answer

st.set_page_config(page_title="Appliance Manual Assistant", page_icon="🧊")
st.title("Appliance Manual Assistant (RAG demo)")
import config as C
hybrid = st.sidebar.toggle("Hybrid search (vector + keyword)", value=True)
rerank = st.sidebar.toggle("Re-rank with cross-encoder (slower, more accurate)", value=bool(C.RERANK))
if "chat" not in st.session_state:
    st.session_state.chat = []
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)
if q := st.chat_input("Ask about the manuals (German or English)…"):
    st.chat_message("user").write(q)
    with st.spinner("Searching manuals…"):
        text, hits = answer(q, hybrid=hybrid, rerank=rerank)
    st.chat_message("assistant").write(text)
    with st.expander("Retrieved sources"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h['source']} – page {h['page']}**\n\n{h['text'][:400]}…")
    st.session_state.chat += [("user", q), ("assistant", text)]
