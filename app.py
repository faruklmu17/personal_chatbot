# app.py
"""
Sleek Gradio UI for the personal chatbot (PROFILE-based).
- Soft theme, glassy cards, quick-question chips
- Responsive layout (sidebar on desktop, stacked on mobile)
- Copy-to-clipboard on messages, clear chat, and starter tips
"""

import os
import pathlib
import subprocess
from typing import Dict, Any, Callable, List

import joblib
import gradio as gr

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
answers_index = packed["answers_index"]
PROFILE: Dict[str, Any] = packed["profile"]

# --- Renderers (mirror train_model.py keys) ---
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

# --- Theme & CSS ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate"
)

custom_css = """
/* Center the app and set max width */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* Header card */
.header-card {
  background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(14,165,233,.10));
  border: 1px solid rgba(255,255,255,.12);
  backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 18px 18px;
}

/* Glass cards */
.glass {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  backdrop-filter: blur(12px) !important;
  border-radius: 18px !important;
}

/* Chatbot bubble tweaks */
.message.bot { background: rgba(99,102,241,.12) !important; border: 1px solid rgba(99,102,241,.25) !important; }
.message.user { background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.15) !important; }

/* Buttons: chips */
.quick-chip button {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  border-radius: 999px !important;
  padding: 8px 14px !important;
}
.quick-chip button:hover { transform: translateY(-1px); }

/* Footer */
.footer {
  opacity: .75; font-size: .9rem; text-align: center; padding: 6px 0;
}
"""

# ---- UI ----
with gr.Blocks(title="Faruk Hasan – Personal Chatbot", theme=theme, css=custom_css) as demo:
    # Header
    with gr.Row(elem_classes=["header-card"]):
        gr.HTML(
            """
            <div style="display:flex;align-items:center;gap:14px;">
              <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#6366f1,#22d3ee);display:flex;align-items:center;justify-content:center;font-size:22px;">🤖</div>
              <div style="display:flex;flex-direction:column;">
                <div style="font-weight:700;font-size:1.2rem;letter-spacing=.2px">Faruk Hasan — Personal Chatbot</div>
                <div style="color:#a5b4fc;font-size:.95rem;">Ask about education, tools, work, tutoring, or personal life.</div>
              </div>
              <div style="margin-left:auto;">
                <span style="background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.4);padding:6px 10px;border-radius:999px;color:#86efac;font-size:.85rem;">Online</span>
              </div>
            </div>
            """
        )

    with gr.Row():
        # Sidebar (Quick Questions)
        with gr.Column(scale=4, min_width=260):
            with gr.Group(elem_classes=["glass"]):
                gr.Markdown("#### 🔎 Quick Questions")
                chips = [
                    gr.Button("Full name", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Where are you from?", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Where do you live?", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Education", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Tutoring career", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Professional experience", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Tools & skills", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Childhood", size="sm", elem_classes=["quick-chip"]),
                    gr.Button("Personal life", size="sm", elem_classes=["quick-chip"]),
                ]
            with gr.Group(elem_classes=["glass"]):
                gr.Markdown("#### 💡 Tips")
                gr.Markdown(
                    "- Try short prompts like **“full name”**, **“degree”**, **“kids”**.\n"
                    "- Answers are drawn from my structured profile for accuracy.\n"
                    "- Use the **Clear** button to start over."
                )

        # Main chat area
        with gr.Column(scale=8, min_width=520):
            with gr.Group(elem_classes=["glass"]):
                chat = gr.Chatbot(
                    label="Chat",
                    height=520,
                    show_copy_button=True,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me something… (e.g., education, tools, where are you from)",
                        scale=8,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", variant="secondary", scale=1)

    gr.HTML('<div class="footer">© 2025 Faruk Hasan — Personal Chatbot</div>')

    # --- Logic bindings ---
    def respond(message, history):
        reply = route_and_answer(message)
        history = history or []
        history.append((message, reply))
        return "", history

    # Buttons push canned prompts into the box then auto-send
    def inject_and_send(prompt, history):
        reply = route_and_answer(prompt)
        history = history or []
        history.append((prompt, reply))
        return history

    msg.submit(respond, [msg, chat], [msg, chat])
    send.click(respond, [msg, chat], [msg, chat])
    clear.click(lambda: ([],), outputs=[chat])

    chips[0].click(lambda h: inject_and_send("full name", h), inputs=[chat], outputs=[chat])
    chips[1].click(lambda h: inject_and_send("where are you from", h), inputs=[chat], outputs=[chat])
    chips[2].click(lambda h: inject_and_send("where do you live", h), inputs=[chat], outputs=[chat])
    chips[3].click(lambda h: inject_and_send("education", h), inputs=[chat], outputs=[chat])
    chips[4].click(lambda h: inject_and_send("tutoring career", h), inputs=[chat], outputs=[chat])
    chips[5].click(lambda h: inject_and_send("professional career", h), inputs=[chat], outputs=[chat])
    chips[6].click(lambda h: inject_and_send("tools and skills", h), inputs=[chat], outputs=[chat])
    chips[7].click(lambda h: inject_and_send("childhood", h), inputs=[chat], outputs=[chat])
    chips[8].click(lambda h: inject_and_send("personal life", h), inputs=[chat], outputs=[chat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
