import io
import os
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from PIL import Image
from pdf2image import convert_from_bytes

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# ===== Model config =====
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")
MODEL_PATH = os.getenv("MODEL_PATH") or MODEL_ID
dtype_env = os.getenv("DTYPE", "bf16").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    DTYPE = torch.float32
else:
    DTYPE = torch.bfloat16 if dtype_env == "bf16" else torch.float16

_is_local_model = os.path.isdir(MODEL_PATH)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=_is_local_model,
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    local_files_only=_is_local_model,
).eval()

app = FastAPI(title="Qwen3-VL Pod")

# In-memory conversation store for demo purposes.
SESSIONS: Dict[str, List[dict]] = {}


def ensure_session(session_id: Optional[str]) -> str:
    """Return an existing session id or create a brand-new one."""
    if not session_id:
        session_id = str(uuid.uuid4())
    SESSIONS.setdefault(session_id, [])
    return session_id


def clone_messages(messages: List[dict]) -> List[dict]:
    """Create a shallow clone of messages so we avoid side effects from model.chat."""
    cloned: List[dict] = []
    for msg in messages:
        new_content = []
        for item in msg.get("content", []):
            if item.get("type") == "image":
                new_content.append({"type": "image", "image": item["image"]})
            else:
                new_content.append({"type": item.get("type", "text"), "text": item.get("text", "")})
        cloned.append({"role": msg.get("role", "user"), "content": new_content})
    return cloned


def gen_from_messages(
    messages: List[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Generate response using Qwen3-VL generation workflow."""
    encoded = processor.apply_chat_template(
        clone_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {"max_new_tokens": int(max_new_tokens)}
    if temperature and float(temperature) > 0:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["do_sample"] = True
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**encoded, **generate_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(encoded["input_ids"], generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs[0] if outputs else ""


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str
    max_new_tokens: int = 512
    temperature: float = 0.2


@app.post("/chat")
def chat(req: ChatReq):
    sid = ensure_session(req.session_id)
    history = SESSIONS[sid]
    history.append({"role": "user", "content": [{"type": "text", "text": req.message}]})
    text = gen_from_messages(history, req.max_new_tokens, req.temperature)
    history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {"session_id": sid, "output": text}


@app.post("/vision-chat")
async def vision_chat(
    message: str = Form(...),
    image: UploadFile = File(None),
    session_id: Optional[str] = Form(None),
    page: int = Form(1),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.2),
):
    sid = ensure_session(session_id)
    history = SESSIONS[sid]

    contents = []
    if image is not None:
        blob = await image.read()
        filename = (image.filename or "").lower()
        is_pdf = filename.endswith(".pdf") or image.content_type == "application/pdf"
        if is_pdf:
            pages = convert_from_bytes(blob, dpi=220)
            if not pages:
                raise ValueError("PDF does not contain any pages.")
            idx = max(0, min(len(pages) - 1, page - 1))
            pil = pages[idx].convert("RGB")
        else:
            pil = Image.open(io.BytesIO(blob)).convert("RGB")
        contents.append({"type": "image", "image": pil})

    contents.append({"type": "text", "text": message})
    history.append({"role": "user", "content": contents})

    text = gen_from_messages(history, int(max_new_tokens), float(temperature))
    history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {"session_id": sid, "output": text}
