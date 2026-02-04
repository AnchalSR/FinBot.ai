import streamlit as st
import requests
import time

# ================= CONFIG =================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FinBot AI",
    page_icon="🤖",
    layout="centered"
)

# ================= CSS =================

st.markdown("""
<style>

.chat-box {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
}

.user-box {
    background-color: #DCF8C6;
    margin-left: auto;
}

.bot-box {
    background-color: #F1F0F0;
    margin-right: auto;
}

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.title("🤖 FinBot AI")
st.caption("Your Smart Financial Assistant")

# ================= SESSION =================

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= DISPLAY CHAT =================

def show_chat():

    for msg in st.session_state.chat:

        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-box user-box'>{msg['text']}</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"<div class='chat-box bot-box'>{msg['text']}</div>",
                unsafe_allow_html=True
            )

show_chat()

# ================= INPUT =================

with st.form("chat_form", clear_on_submit=True):

    query = st.text_input(
        "Type your question...",
        placeholder="Ask about investment, tax, finance..."
    )

    send = st.form_submit_button("Send 🚀")

# ================= CHAT LOGIC =================

if send and query.strip():

    # Add user msg
    st.session_state.chat.append({
        "role": "user",
        "text": query
    })

    # Loading
    with st.spinner("FinBot is thinking... 🤔"):
        time.sleep(1)

        payload = {
            "query": query,
            "top_k": 5,
            "include_sources": True
        }

        try:
            res = requests.post(
                f"{API_URL}/chat",
                json=payload,
                timeout=30
            )

            if res.status_code == 200:
                answer = res.json().get("answer", "No response received.")
            else:
                try:
                    error_detail = res.json().get("detail", f"Server returned {res.status_code}")
                except:
                    error_detail = f"Server error {res.status_code}"
                
                if res.status_code == 503:
                    answer = f"⚠️ Backend not ready (503): {error_detail}\n\n💡 Tip: Check if documents are uploaded or RAG initialized."
                else:
                    answer = f"⚠️ Error ({res.status_code}): {error_detail}"

        except requests.exceptions.Timeout:
            answer = "❌ Request timeout. Backend may be slow or RAG not initialized."
        except requests.exceptions.ConnectionError:
            answer = "❌ Cannot connect to backend. Run: python -m finbot.backend.api"
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    # Add bot msg
    st.session_state.chat.append({
        "role": "bot",
        "text": answer
    })

    st.rerun()

# ================= SIDEBAR =================

st.sidebar.markdown(
    "<div class='sidebar-title'>📂 Document Upload</div>",
    unsafe_allow_html=True
)

files = st.sidebar.file_uploader(
    "Upload PDF / TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Upload Files"):

    if files:

        upload_data = []

        for file in files:
            upload_data.append(
                ("files", (file.name, file.read(), file.type))
            )

        try:
            res = requests.post(
                f"{API_URL}/upload",
                files=upload_data
            )

            if res.status_code == 200:
                st.sidebar.success("✅ Uploaded Successfully")
            else:
                st.sidebar.error("❌ Upload Failed")

        except:
            st.sidebar.error("❌ Server Not Reachable")

    else:
        st.sidebar.warning("Select files first")

# ================= CLEAR =================

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.chat = []
    st.rerun()


# ================= FOOTER =================

st.markdown("---")
st.caption("🚀 Built with FastAPI + Streamlit | FinBot AI")
