import os
import requests
import streamlit as st

POD_URL = os.getenv("POD_URL", "http://<POD_PUBLIC_IP>:8000")

st.set_page_config(page_title="Qwen3-VL Chat (Pod Mode A)", layout="wide")
st.title("🧱 Qwen3-VL Chat • Mode A (Client UI → Pod HTTP)")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Settings")
    max_new_tokens = st.slider("max_new_tokens", 64, 1024, 512, 64)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption(f"Pod URL: {POD_URL}")
    reset = st.button("🔁 New session")
    if reset:
        st.session_state.session_id = None
        st.session_state.history = []
        st.rerun()

# Render existing history
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

uploaded = st.file_uploader(
    "Upload image/PDF (optional)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=False,
)
user_msg = st.chat_input("Nhập câu hỏi… (VD: 'Hãy liệt kê các phòng và diện tích')")


def call_text_chat(message: str) -> str:
    response = requests.post(
        f"{POD_URL}/chat",
        json={
            "session_id": st.session_state.session_id,
            "message": message,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    st.session_state.session_id = data["session_id"]
    return data["output"]


def call_vision_chat(message: str, fileobj) -> str:
    files = None
    data = {
        "message": message,
        "session_id": st.session_state.session_id or "",
        "max_new_tokens": str(max_new_tokens),
        "temperature": str(temperature),
    }
    if fileobj is not None:
        file_bytes = fileobj.read()
        files = {"image": (fileobj.name, file_bytes, fileobj.type or "application/octet-stream")}
    response = requests.post(
        f"{POD_URL}/vision-chat",
        data=data,
        files=files,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    st.session_state.session_id = data["session_id"]
    return data["output"]


if user_msg is not None:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ…"):
            if uploaded is not None:
                uploaded.seek(0)
                output = call_vision_chat(user_msg, uploaded)
            else:
                output = call_text_chat(user_msg)
            st.markdown(output)
    st.session_state.history.append(("assistant", output))
    st.rerun()
