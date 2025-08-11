# app.py
"""
Gradio app for the personal chatbot (structured PROFILE-based).

- Ensures artifacts exist on startup (runs train_model.py if missing)
- Loads classifier (intent router) + PROFILE
- Renders answers from PROFILE via deterministic renderers
"""

import os
import pathlib
import subprocess
import joblib
import gradio as gr
from typing import Dict, Any, Callable, List

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")
ANSWERS_PATH = os.getenv("ANSWERS_PATH", "answers.pkl")


def ensure_artifacts():
    need = [MODEL_PATH, VECTORIZER_PATH, ANSWERS_PATH]
    if not all(pathlib.Path(p).exists() for p in need):
        print("[INFO] Artifacts missing — training model...")
        subprocess.run(["python", "train_model.py"], check=True)
        print("[INFO] Training finished.")


ensure_artifacts()

# --- Load artifacts ---
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
packed = joblib.load(ANSWERS_PATH)
answers_index = packed["answers_index"]          # intent -> renderer key (same name)
PROFILE: Dict[str, Any] = packed["profile"]      # structured profile

# --- Renderers (must match keys used in train_model.py) ---
def render_full_name(p: Dict[str, Any]) -> str:
    return f"My full name is {p.get('full_name', '—')}."

def render_origin(p: Dict[str, Any]) -> str:
    bp = p.get("birthplace")
    if bp:
        return f"I was born in {bp}."
    origin = p.get("origin")
    return f"I’m originally from {origin}." if origin else "I’m originally from —."

def render_location(p: Dict[str, Any]) -> str:
    loc = p.get("current_location")
    return f"I currently live in {loc}." if loc else "I currently live in —."

def render_education(p: Dict[str, Any]) -> str:
    edu: List[Dict[str, str]] = p.get("education", [])
    if not edu:
        return "Education: (add items in PROFILE['education'])."
    lines = []
    for e in edu:
        inst = e.get("institution", "Institution")
        deg  = e.get("degree", "").strip()
        fld  = e.get("field", "").strip()
        yrs  = e.get("years", "").strip()
        notes = e.get("notes", "").strip()
        parts = [v for v in [deg, fld, yrs] if v]
        suffix = " — " + " • ".join(parts) if parts else ""
        if notes:
            suffix += f" • {notes}"
        lines.append(f"- {inst}{suffix}")
    return "Education:\n" + "\n".join(lines)

def render_tutoring(p: Dict[str, Any]) -> str:
    t = p.get("tutoring_career", {})
    parts = []
    if t.get("summary"): parts.append(t["summary"])
    if t.get("since"): parts.append(f"Teaching since {t['since']}.")
    if t.get("topics"): parts.append("Topics: " + ", ".join(t["topics"]) + ".")
    if t.get("platforms"): parts.append("Platforms: " + ", ".join(t["platforms"]) + ".")
    return " ".join(parts) or "I teach coding/AI courses."

def render_professional(p: Dict[str, Any]) -> str:
    jobs: List[Dict[str, Any]] = p.get("professional_experience", [])
    if not jobs:
        return "Professional experience: (add items in PROFILE['professional_experience'])."
    out = ["Professional Experience:"]
    for j in jobs:
        title = j.get("title", "Role")
        comp  = j.get("company", "Company")
        yrs   = j.get("years", "")
        line1 = f"- {title} @ {comp}" + (f" ({yrs})" if yrs else "")
        out.append(line1)
        for h in j.get("highlights", [])[:4]:
            out.append(f"   • {h}")
    return "\n".join(out)

def render_tools(p: Dict[str, Any]) -> str:
    s = p.get("tools_and_skills", {})
    buckets = []
    for key in ["languages", "testing", "devops", "cloud", "data_ai", "other"]:
        vals = s.get(key)
        if vals:
            label = key.replace("_", " ").title()
            buckets.append(f"{label}: " + ", ".join(vals))
    return " | ".join(buckets) if buckets else "Skills: (add lists in PROFILE['tools_and_skills'])."

def render_childhood(p: Dict[str, Any]) -> str:
    return p.get("childhood", "I grew up loving technology and problem‑solving.")

def render_personal(p: Dict[str, Any]) -> str:
    info = p.get("personal_life", {})
    parts = []
    if info.get("family"):
        parts.append(info["family"])
    if info.get("hobbies"):
        parts.append("Hobbies: " + ", ".join(info["hobbies"]) + ".")
    if info.get("fun_facts"):
        parts.append("Fun facts: " + "; ".join(info["fun_facts"]) + ".")
    return " ".join(parts) or "I enjoy family time, travel, and teaching."

RENDERERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "greeting": lambda p: "Hi! Ask about my education, tools, work, tutoring, or personal life.",
    "help": lambda p: ("You can ask about my full name, where I’m from/born, where I live, "
                       "my education, tutoring career, professional experience, tools/skills, childhood, or personal life."),
    "thanks": lambda p: "You’re welcome!",
    "full_name": render_full_name,
    "origin": render_origin,
    "current_location": render_location,
    "education": render_education,
    "tutoring_career": render_tutoring,
    "professional_career": render_professional,
    "tools_and_skills": render_tools,
    "childhood": render_childhood,
    "personal_life": render_personal,
}

def route_and_answer(user_text: str) -> str:
    X = vectorizer.transform([user_text])
    intent = model.predict(X)[0]
    key = answers_index.get(intent, "help")
    renderer = RENDERERS.get(key, RENDERERS["help"])
    return renderer(PROFILE)

# ---- Gradio UI ----
with gr.Blocks(title="Faruk Hasan - Personal Chatbot") as demo:
    gr.Markdown("## 🤖 Ask me about my background!\nTry: **full name**, **where are you from**, **education**, **tools**, **kids**")
    chat = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ask me something…", label="Your question")

    def respond(message, history):
        reply = route_and_answer(message)
        history.append((message, reply))
        return "", history

    msg.submit(respond, [msg, chat], [msg, chat])

if __name__ == "__main__":
    # HF handles host/port automatically; these are fine for local dev.
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
