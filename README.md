title: Faruk Hasan – Personal Chatbot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
tags:
  - chatbot
  - personal
  - portfolio
  - gradio
  - resume
  - qna
---

# 🤖 Faruk Hasan – Personal Chatbot

A lightweight **Gradio** chatbot that answers questions about **Faruk Hasan**—covering tutoring career, professional experience, education, tools/skills, childhood, and personal life.  
Built with **Python**, **scikit‑learn (Naive Bayes)**, and **Gradio**. No external LLMs or API keys required.

You can ask things like:
- “What is your full name?”
- “Where are you from and where do you live?”
- “What’s your education background?”
- “Tell me about your tutoring career.”
- “Which tools and languages do you use?”
- “Do you have kids? What are your hobbies?”

---

## 📦 Features

- ✅ Intent-based Q&A (Naive Bayes) with short and long phrasing support (e.g., “full name”, “degree”, “kids”)
- ✅ Clean **Gradio** web UI
- ✅ Zero external services — fully offline-capable
- ✅ Easy to embed on any site via `<iframe>`
- ✅ Simple to extend: add new intents & answers in one file

---

## 🚀 Getting Started

1) **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Train the tiny intent model**

```bash
python train_model.py
```

This creates:

* `model.pkl`
* `vectorizer.pkl`
* `answers.pkl`

3. **Run the app**

```bash
python app.py
```

Open `http://127.0.0.1:7860` and start chatting.

> Tip: On Hugging Face Spaces, commit the generated `.pkl` files so the app loads instantly.
> (Alternatively, add a startup train step if you prefer training on deploy.)

---

## 🔍 How It Works

1. **Intents & Phrases**
   In `train_model.py`, `TRAIN_DEFAULTS` lists intents (e.g., `full_name`, `origin`, `education`, `tutoring_career`, `professional_career`, `tools_and_skills`, `childhood`, `personal_life`) with multiple example phrasings, including short forms like “full name”, “kids”, “degree”.

2. **Training**
   We vectorize text with `CountVectorizer` (1–2 n‑grams) and train a `MultinomialNB` classifier.
   Outputs are saved to `model.pkl`, `vectorizer.pkl`, and `answers.pkl`.

3. **Serving with Gradio**
   `app.py` loads the artifacts and routes user input to the classifier, returning the corresponding canned answer from `answers.pkl`.

---

## 🧱 Project Structure

```
.
├── app.py             # Gradio app (loads artifacts, runs chatbot)
├── train_model.py     # Intents, answers, training script
├── requirements.txt   # Dependencies (pinned)
└── README.md          # This file
```

(After training, you’ll also have `model.pkl`, `vectorizer.pkl`, `answers.pkl`.)

---

## 🌐 Embed on Your Website (iframe)

### Minimal embed

```html
<iframe
  src="https://huggingface.co/spaces/<your-username>/<space-name>?embed=true"
  style="width:100%;height:600px;border:0;"
  allow="clipboard-read; clipboard-write; microphone">
</iframe>
```

### Bottom‑right floating widget (toggle)

```html
<button id="fh-chat-toggle"
  style="position:fixed;right:22px;bottom:22px;z-index:99999;padding:10px 14px;border-radius:999px;border:none;box-shadow:0 6px 20px rgba(0,0,0,.15);cursor:pointer;">
  Chat
</button>

<div id="fh-chat-container"
  style="position:fixed;right:22px;bottom:82px;width:380px;height:560px;max-width:90vw;max-height:80vh;display:none;z-index:99998;">
  <iframe
    id="fh-chat-iframe"
    src="https://huggingface.co/spaces/<your-username>/<space-name>?embed=true"
    style="width:100%;height:100%;border:0;border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.25);"
    allow="clipboard-read; clipboard-write; microphone">
  </iframe>
</div>

<script>
  const btn = document.getElementById('fh-chat-toggle');
  const box = document.getElementById('fh-chat-container');
  let open = false;
  btn.addEventListener('click', () => {
    open = !open;
    box.style.display = open ? 'block' : 'none';
    btn.textContent = open ? 'Close' : 'Chat';
  });
</script>
```

---

## 🛠 Dependencies

Install via:

```bash
pip install -r requirements.txt
```

Pinned (example):

* `gradio==4.44.0`
* `scikit-learn==1.7.1`
* `joblib==1.4.2`

Optional: add `runtime.txt` with `python-3.10` (or 3.11) for consistent HF builds.

---

## ❓FAQ

* **Will short queries like “full name” or “kids” work?**
  Yes—`TRAIN_DEFAULTS` includes minimal keywords and variants so short inputs classify correctly.

* **How do I add more topics?**
  Add a new intent with training phrases and a corresponding answer in `train_model.py`, then re-run `python train_model.py`.

* **Private details?**
  Only include what you’re comfortable sharing. You control all answers in `DEFAULT_ANSWERS`.

---

## 👨‍💻 Author

Created by **Faruk Hasan**
Senior QA Engineer | Coding Instructor | AI Explorer

---

## 📄 License

MIT License

```
```
