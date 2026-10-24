"""AskMyDocs UI. Talks to the FastAPI backend only over HTTP (API_URL, default http://localhost:8000).

Run:  streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
FILE_TYPES = ["pdf", "docx", "epub", "txt", "md", "png", "jpg", "jpeg"]
STATUS_ICON = {"processing": "⏳", "ready": "✅", "failed": "❌"}
ROUTE_LABEL = {
    "specific": "searched the text",
    "whole_document": "used the document map (chapter summaries)",
    "entity": "used the character / entity index",
}

st.set_page_config(page_title="AskMyDocs", page_icon="📚", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the backend; return (json or None, error message or None)."""
    try:
        response = requests.request(method, f"{API}{path}", timeout=180, **kwargs)
    except requests.ConnectionError:
        return None, f"Can't reach the backend at {API}. Is `uvicorn app.main:app` running?"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, str(detail)
    return (response.json() if response.content else {}), None


# ---------- sidebar: upload + document list ----------
with st.sidebar:
    st.title("📚 AskMyDocs")
    st.caption("Answers only from your document, with citations.")

    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("Upload a document", type=FILE_TYPES)
        sensitive = st.checkbox(
            "🔒 Sensitive mode",
            help="For offer letters, agreements, statements: names, IDs, phone numbers etc. are replaced "
            "with placeholders on this computer before anything is sent to the AI. Leave off for books.",
        )
        if st.form_submit_button("Upload", width="stretch") and uploaded:
            body, error = api(
                "POST",
                "/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={"sensitive": str(sensitive).lower()},
            )
            if error:
                st.error(error)
            else:
                st.session_state["selected"] = body["id"]
                if body.get("duplicate"):
                    st.info("This exact file was already uploaded, so I reused it.")

    documents, error = api("GET", "/documents")
    if error:
        st.error(error)
        st.stop()
    if not documents:
        st.info("Upload a PDF, DOCX, EPUB, TXT, MD or an image to start.")
        st.stop()

    ids = [d["id"] for d in documents]
    by_id = {d["id"]: d for d in documents}
    if st.session_state.get("selected") not in ids:
        st.session_state["selected"] = ids[0]
    selected = st.radio(
        "Your documents",
        ids,
        key="selected",
        format_func=lambda i: (
            f"{STATUS_ICON.get(by_id[i]['status'], '')} {'🔒 ' if by_id[i]['sensitive'] else ''}{by_id[i]['filename']}"
        ),
    )


# ---------- main area ----------
@st.fragment(run_every=2)
def status_panel(doc_id: int):
    """Re-runs every 2 s while something is in progress, so progress updates live."""
    doc, error = api("GET", f"/documents/{doc_id}")
    if error:
        st.error(error)
        return
    busy = doc["status"] == "processing" or doc["map_status"] == "building"
    if doc["status"] == "processing":
        st.info(f"⏳ Processing: {doc['progress'] or 'starting'}")
    elif doc["status"] == "failed":
        st.error(f"❌ {doc['error']}")
    else:
        bits = [f"{doc['n_sections']} sections", f"{doc['n_chunks']} chunks"]
        if doc["n_pages"]:
            bits.insert(0, f"{doc['n_pages']} pages")
        if doc["ocr_pages"]:
            bits.append(f"{doc['ocr_pages']} page(s) read with OCR")
        st.caption(" · ".join(bits))
        if doc["map_status"] == "building":
            st.info(f"🗺️ Building document map: {doc['progress'] or ''}")
        elif doc["map_status"] in {"failed", "pending"} and doc["progress"]:
            st.warning(doc["progress"])
    if not busy and st.session_state.get(f"was_busy_{doc_id}"):
        st.session_state[f"was_busy_{doc_id}"] = False
        st.rerun()  # finished: refresh the whole page once (chat box, map tab)
    st.session_state[f"was_busy_{doc_id}"] = busy


doc = by_id[selected]
st.header(
    ("🔒 " if doc["sensitive"] else "") + (doc["filename"] if doc["sensitive"] else doc["title"] or doc["filename"])
)
status_panel(selected)
show_redacted = doc["sensitive"] and st.toggle(
    "Show what the AI saw (placeholders instead of real values)",
    help="In sensitive mode the AI only ever receives placeholders like [PERSON_1]; real values are "
    "put back on this computer just before showing you the answer.",
)

chat_tab, map_tab, manage_tab = st.tabs(["💬 Chat", "🗺️ Document map", "⚙️ Manage"])

with chat_tab:
    history = st.session_state.setdefault("chats", {}).setdefault(selected, [])
    for message in history:
        with st.chat_message(message["role"]):
            redacted_view = show_redacted and message.get("answer_redacted")
            st.markdown(message["answer_redacted"] if redacted_view else message["content"])
            if message["role"] == "assistant":
                if message.get("route"):
                    st.caption(f"Route: {ROUTE_LABEL.get(message['route'], message['route'])}")
                for c in message.get("citations", []):
                    with st.expander(f"📎 {c['source']}, {c['loc']}"):
                        st.write(c.get("text_redacted", c["text"]) if show_redacted else c["text"])
                if message.get("sources"):
                    with st.expander(f"All {len(message['sources'])} excerpts the AI was shown"):
                        for s in message["sources"]:
                            st.markdown(f"**{s['loc']}** ({s['kind'].replace('_', ' ')})")
                            st.write(s.get("text_redacted", s["text"]) if show_redacted else s["text"])
                if message.get("llm_payload"):
                    with st.expander("🔍 What was sent to the AI (exact request)"):
                        for m in message["llm_payload"]:
                            st.markdown(f"**{m['role']}**")
                            st.code(m["content"], language=None, wrap_lines=True)

    if doc["status"] != "ready":
        st.chat_input("The document isn't ready yet", disabled=True)
    elif question := st.chat_input("Ask anything about this document"):
        history.append({"role": "user", "content": question})
        with st.spinner("Searching the document..."):
            answer, error = api("POST", f"/documents/{selected}/ask", json={"question": question})
        if error:
            history.append({"role": "assistant", "content": f"⚠️ {error}"})
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": answer["answer"],
                    "route": answer["route"],
                    "citations": answer["citations"],
                    "sources": answer["sources"],
                    "answer_redacted": answer.get("answer_redacted"),
                    "llm_payload": answer.get("llm_payload"),
                }
            )
        st.rerun()

with map_tab:
    docmap, error = api("GET", f"/documents/{selected}/map", params={"redacted": str(show_redacted).lower()})
    if error:
        st.error(error)
    else:
        if docmap["map_status"] not in {"ready", "building"} and doc["status"] == "ready":
            started = any(s["summary"] for s in docmap["sections"])
            if st.button("Resume building the map" if started else "Build map"):
                _, error = api("POST", f"/documents/{selected}/map")
                if error:
                    st.error(error)
                else:
                    st.rerun()
        if docmap["summary"]:
            st.subheader("Overview")
            st.write(docmap["summary"])
            name = doc["filename"].rsplit(".", 1)[0]
            if doc["sensitive"]:
                safe, _ = api("GET", f"/documents/{selected}/map", params={"redacted": "true"})
                st.download_button(
                    "⬇️ Summary with placeholders (safe to share)", safe["summary"], f"{name}-summary-redacted.txt"
                )
            st.download_button("⬇️ Summary", docmap["summary"], f"{name}-summary.txt")
        if docmap["entities"]:
            st.subheader("Characters & entities")
            st.dataframe(
                [
                    {
                        "Name": e["name"],
                        "Kind": e["kind"],
                        "Also called": ", ".join(e["aliases"]),
                        "Who / what": e["description"],
                        "Appears in": ", ".join(e["appears_in"]),
                    }
                    for e in docmap["entities"]
                ],
                width="stretch",
                hide_index=True,
            )
        st.subheader(f"Sections ({len(docmap['sections'])})")
        for s in docmap["sections"]:
            pages = f" · pages {s['page_start']}-{s['page_end']}" if s["page_start"] else ""
            with st.expander(f"{s['label']} — {s['title']}{pages} · {s['n_words']} words"):
                st.write(s["summary"] or "_Not summarised yet._")

with manage_tab:
    st.write(f"**File:** {doc['filename']} · **Type:** {doc['file_type']} · **Uploaded:** {doc['created_at'][:16]}")
    if doc["sensitive"] and doc["status"] == "ready":
        report, error = api("GET", f"/documents/{selected}/pii-report")
        st.subheader("🔒 Personal data found (hidden from the AI)")
        if error:
            st.error(error)
        elif not report["types"]:
            st.info("No personal data was detected.")
        else:
            st.dataframe(
                [
                    {"Type": t["type"], "Different values": t["distinct"], "Times it appears": t["occurrences"],
                     "Where": ", ".join(t["where"])}
                    for t in report["types"]
                ],
                width="stretch",
                hide_index=True,
            )  # fmt: skip
            st.caption(
                "Detection is best-effort: unusual formats, nicknames or addresses can slip through. "
                "Before every request the tripwire re-checks for these values and blocks anything that leaks."
            )
    st.write("Deleting removes the document's text, embeddings, summaries, entities, PII vault and question log.")
    confirm = st.checkbox("Yes, delete this document permanently")
    if st.button("🗑️ Delete document", type="primary", disabled=not confirm):
        _, error = api("DELETE", f"/documents/{selected}")
        if error:
            st.error(error)
        else:
            st.session_state.get("chats", {}).pop(selected, None)
            st.session_state.pop("selected", None)
            st.rerun()
