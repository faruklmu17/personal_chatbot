# train_model.py
"""
Structured-personal-bot trainer:
- Stores your info in a PROFILE dict (EDIT below if needed)
- Builds intent classifier (Naive Bayes) for routing
- Renders answers deterministically from PROFILE

Outputs:
  model.pkl
  vectorizer.pkl
  answers.pkl  (maps intent -> renderer key + ships PROFILE)
"""

import joblib
from typing import Dict, Any, Iterable, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------
# 1) YOUR PROFILE (filled from your about_me HTML)
# ---------------------------
PROFILE: Dict[str, Any] = {
    "full_name": "Faruk Hasan",
    "birth_year": 1988,  # from “born in 1988”
    "birthplace": "Chandpur, Bangladesh",
    "origin": "Bangladesh",
    "current_location": "Hudson Valley, New York, USA",  # “living in the Hudson Valley for the past three years”
    "cities_lived": ["Dhaka", "London", "Los Angeles", "New York City"],

    # Education (newest → oldest)
    "education": [
        {
            "institution": "Loyola Marymount University (Los Angeles, USA)",
            "degree": "Master’s",
            "field": "",  # field not specified in HTML
            "years": "2014–2016",
            "notes": "Completed with GPA 3.54"
        },
        {
            "institution": "Queen Mary University of London (UK)",
            "degree": "B.Eng",
            "field": "",  # electrical track implied prior, but HTML doesn’t explicitly state here
            "years": "2009–2012",
            "notes": "Upper Second Class"
        },
        {
            "institution": "American International University–Bangladesh (AIUB)",
            "degree": "Electrical & Electronics Engineering (transfer)",
            "field": "EEE",
            "years": "2007–2009",
            "notes": "Transferred credits to QMUL"
        },
        {
            "institution": "Udayan Higher Secondary School (Dhaka)",
            "degree": "HSC",
            "field": "Science",
            "years": "2004–2006",
            "notes": ""
        },
        {
            "institution": "Government Science Attached High School (Tejgaon)",
            "degree": "SSC",
            "field": "",
            "years": "2001–2004",
            "notes": "SSC in 2004, GPA 4.81/5"
        },
        {
            "institution": "Alim Uddin High School",
            "degree": "Grades 6–7",
            "field": "",
            "years": "1999–2001",
            "notes": ""
        },
        {
            "institution": "Little Flower School (Mirpur, Dhaka)",
            "degree": "Grades 3–5",
            "field": "",
            "years": "1996–1999",
            "notes": "Top student in Grade 5; “Good Student” title"
        },
        {
            "institution": "Sylhet (early schooling)",
            "degree": "Up to Grade 2",
            "field": "",
            "years": "1994–1996",
            "notes": ""
        },
    ],

    # Tutoring / Teaching
    "tutoring_career": {
        "summary": "Independent technology instructor teaching middle and high school students (project‑based).",
        "topics": [
            "Python, Java, HTML/CSS/JavaScript, React",
            "AI/ML basics: chatbots, Bag‑of‑Words, Naive Bayes",
            "Web dev projects, Git/GitHub, deployment"
        ],
        "platforms": ["Outschool", "Private workshops/mentoring"],
        "since": "2016"  # teaching roles begin post‑Master’s per timeline
    },

    # Professional experience (newest → oldest)
    "professional_experience": [
        {
            "title": "Senior QA Engineer",
            "company": "(Remote)",
            "years": "2020–2025",
            "highlights": [
                "Grew into Senior QA role; focus on automation and quality",
                "Playwright/Selenium automation and CI/CD integration",
                "Ongoing educator on Outschool alongside QA work"
            ]
        },
        {
            "title": "QA Engineer",
            "company": "(Albany, New York)",
            "years": "2020",
            "highlights": [
                "First tech role in QA; relocation from NYC to Albany"
            ]
        },
        {
            "title": "Educator / Lecturer (STEM) & Other Roles",
            "company": "Charter schools / Community college / Uber",
            "years": "2016–2020",
            "highlights": [
                "Taught science and math; lectured at community college",
                "Various roles while navigating visa/job market",
                "Developed resilience and adaptability"
            ]
        }
    ],

    # Tools & skills
    "tools_and_skills": {
        "languages": ["Python", "Java", "JavaScript"],
        "testing": ["Playwright", "Selenium", "pytest"],
        "devops": ["Git/GitHub", "CI/CD", "Linux"],
        "cloud": ["AWS"],
        "data_ai": ["Pandas", "scikit‑learn", "Naive Bayes (basics)"],
        "other": ["SQL", "Docker (basics)"]
    },

    # Childhood
    "childhood": (
        "Born in Chandpur; moved due to father’s Biman Bangladesh Airlines job. "
        "Early years across Sylhet and Dhaka; extensive travel from a young age."
    ),

    # Personal life / interests
    "personal_life": {
        "family": "Married, with twin kids.",
        "hobbies": ["Teaching", "Writing", "Travel", "Exploring new technologies", "Fitness walks"],
        "fun_facts": [
            "Visited ~14 countries; lived in 3 countries",
            "Enjoy building kid‑friendly AI projects and courses"
        ]
    },

    # (Optional) Travel snapshot from the page
    "travel_countries": [
        "India", "Thailand", "Macau", "Nepal", "Singapore", "Saudi Arabia",
        "United Kingdom", "Italy", "Netherlands", "France",
        "Switzerland", "Belgium", "United Arab Emirates (Dubai)", "Canada"
    ],
    "stats": {
        "countries_visited": 14,
        "years_of_life": 37,
        "years_in_tech_approx": "5+",
        "countries_lived_in": 3
    }
}

# -------------------------------------------------
# 2) INTENTS & TRAINING PHRASES (expand anytime)
# -------------------------------------------------
TRAIN_DEFAULTS: Dict[str, Dict[str, Iterable[str]]] = {
    "greeting": {"x": ["hi", "hello", "hey", "good morning", "good evening", "how are you"]},
    "help": {"x": ["help", "what can you do", "commands", "how to use this"]},
    "thanks": {"x": ["thanks", "thank you", "thx"]},

    "full_name": {"x": [
        "what is your full name", "tell me your name", "who are you",
        "what should I call you", "full name", "name"
    ]},
    "origin": {"x": [
        "where are you from", "which country are you from", "your birthplace", "origin", "born where"
    ]},
    "current_location": {"x": [
        "where do you live", "current location", "city you live in", "where are you based", "location"
    ]},
    "education": {"x": [
        "what is your education background", "tell me about your education", "education", "degree",
        "university", "college", "where did you study"
    ]},
    "tutoring_career": {"x": [
        "tell me about your tutoring career", "what do you teach", "teaching", "tutor", "courses you teach",
        "how long have you been teaching"
    ]},
    "professional_career": {"x": [
        "tell me about your professional career", "what do you do for work", "where do you work",
        "job experience", "profession", "resume"
    ]},
    "tools_and_skills": {"x": [
        "what tools do you know", "skills", "tech stack", "programming languages",
        "software you use", "tools", "technologies"
    ]},
    "childhood": {"x": [
        "tell me about your childhood", "how was your childhood", "early life", "early days"
    ]},
    "personal_life": {"x": [
        "tell me about your personal life", "are you married", "do you have kids",
        "hobbies", "interests", "family", "kids"
    ]},
}

# ---------------------------------------
# 3) RENDERERS: PROFILE -> Nice sentences
# ---------------------------------------
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

RENDERERS = {
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

# ---------------------------------------------------
# 4) Train classifier (intent routing) and dump data
# ---------------------------------------------------
def build_training_corpus(intents: Dict[str, Dict[str, Iterable[str]]]):
    X, y = [], []
    for label, obj in intents.items():
        for phrase in obj["x"]:
            X.append(phrase)
            y.append(label)
    return X, y

def train_and_dump(
    model_path="model.pkl",
    vectorizer_path="vectorizer.pkl",
    answers_path="answers.pkl"
):
    X, y = build_training_corpus(TRAIN_DEFAULTS)
    vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")
    Xv = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(Xv, y)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Store renderer keys + the full PROFILE (so app serves from structured data)
    joblib.dump(
        {"answers_index": {k: k for k in RENDERERS.keys()}, "profile": PROFILE},
        answers_path
    )
    print("Saved:", model_path, vectorizer_path, answers_path)

if __name__ == "__main__":
    train_and_dump()
