---
title: ThriveSpace
emoji: 🌿
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ThriveSpace — AI-Assisted Wellness & Self-Reflection Platform

ThriveSpace is a privacy-first, local-first wellness and self-reflection web application built with Python Flask, local NLP models, a Random Forest wellness-risk prediction pipeline, goal and habit tracking, and curated Indian emergency and mental health resources.

---

## 🌟 Core Features

1. **Authentication & Data Isolation**:
   - Secure user registration, Bcrypt password hashing, session management via Flask-Login.
   - Strict multi-tenant data ownership isolation across all models.
   - Comprehensive CSRF protection across all state-changing operations via Flask-WTF.

2. **Reflective Journaling with AI NLP**:
   - Private journaling with real-time sentiment analysis via TextBlob polarity scores.
   - Six-emotion classification (`joy`, `sadness`, `anger`, `fear`, `surprise`, `love`) powered by a local Hugging Face transformer model (`j-hartmann/emotion-english-distilroberta-base`).
   - Non-clinical, educational presentation of reflection signals.

3. **Daily Mood Tracking & Analytics**:
   - 1–5 mood check-in scale with categorical tags and optional contextual notes.
   - Daily uniqueness with clean update capability.
   - 7-day rolling mood analytics and Chart.js trend visualization anchored to `Asia/Kolkata` local timezone.

4. **Random Forest Wellness Risk Assessment (`/burnout`)**:
   - Evaluates self-reported lifestyle factors (sleep, workload, stress, physical activity) combined with derived journal and mood reflection signals.
   - Preprocessing with `SimpleImputer(strategy='median')`: Missing personal signals are handled by the model's preprocessing pipeline using median values learned from the synthetic training dataset.
   - Classifies wellness risk into Low, Moderate, and Elevated strain categories (Score 0–100).
   - Generates transparent, factor-level explanations comparing user inputs to reference restorative benchmarks.

5. **Explainable Wellness Suggestions & Goals/Habits (`/goals`)**:
   - 100% deterministic, rule-based recommendation engine prioritizing active strain signals.
   - Strict ceiling of 3 suggestions, each displaying actionable advice, grounded rationale, and source signal badges.
   - Goal tracking with progress percentages, target values, and completion tracking.
   - Daily and weekly habit tracking with idempotent local-date check-ins.

6. **Professional Support & Wellness Resources (`/support`)**:
   - Publicly accessible without requiring login for immediate emergency access.
   - Calming Emergency Safety Card with direct access to national helplines (`112`, Tele-MANAS `14416`, KIRAN `1800-599-0019`).
   - Curated directory of verified Indian mental health organizations and student resources.
   - User wellness data and AI/ML inference remain local; external links are provided only for support resources.

---

## 🛠️ Technology Stack

* **Backend**: Python 3.10+, Flask, Jinja2, standard library `zoneinfo`
* **Database**: SQLite, SQLAlchemy ORM
* **Security**: Flask-Bcrypt, Flask-WTF (CSRF), Python-dotenv
* **Machine Learning & NLP**:
  * Scikit-learn (`RandomForestClassifier`, `SimpleImputer`)
  * Hugging Face Transformers & PyTorch (`j-hartmann/emotion-english-distilroberta-base`)
  * TextBlob (Rule-based Sentiment Polarity)
  * Joblib (Pipeline serialization)
* **Frontend**: Vanilla HTML5, CSS3, JavaScript, Chart.js

---

## 🔬 ML Architecture & Limitations

* **Model**: Scikit-learn `RandomForestClassifier` (100 estimators, max depth 8, min samples split 4, min samples leaf 2, random state 42).
* **Evaluation**: Trained and tested on an 80/20 stratified split (1,600 training, 400 test samples).
* **Dataset Notice**: The model was trained on a clearly labeled synthetic demonstration dataset using a transparent rule-based target-generation strategy. Feature importances indicate which variables contributed most during prototype model training; they do not establish individual clinical causation.

---

## 🛡️ Responsible AI & Health Disclaimer

> **Important Notice**: ThriveSpace is an educational, AI-assisted self-reflection and personal wellness tool. It is **not** a medical device, clinical diagnostic tool, or substitute for professional healthcare or psychotherapy.
>
> If you are experiencing chronic exhaustion, severe anxiety, persistent depression, or thoughts of self-harm, please connect immediately with a licensed mental health professional or contact national emergency services at **112** or Tele-MANAS at **14416**.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/thrive-spaces.git
cd thrive-spaces
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Generate and configure a secure secret key:
```env
SECRET_KEY=your_generated_secret_key_here
FLASK_DEBUG=False
APP_TIMEZONE=Asia/Kolkata
```

### 4. Run the Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧪 Running Automated Tests

Run the complete test suite across all feature phases:
```bash
python -m unittest test_phase2.py test_phase3.py test_phase4.py test_phase5.py test_phase6.py
```
Expected output: **57 test methods passing (`Ran 57 tests ... OK`)**.

Run the end-to-end simulated user journey demo:
```bash
python scratch/e2e_demo_test.py
```
Expected output: **All 16 demo steps validated successfully**.

---

## 📋 Known Limitations

* **Technical Prototype**: The Random Forest model demonstrates pipeline integration and missing-signal imputation; it is not clinically validated for diagnostic use.
* **Single-Node SQLite**: Suitable for development and demonstration purposes; multi-user production deployments should transition to PostgreSQL.
* **No Real-time Messaging**: Crisis resources provide verified direct helpline phone numbers and official websites rather than live chat consultations.
