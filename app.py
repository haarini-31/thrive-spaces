import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta

from flask import Flask, render_template, request, redirect, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_wtf.csrf import CSRFProtect, CSRFError
from dotenv import load_dotenv
import requests
from textblob import TextBlob
from transformers import pipeline

from ml.predict_wellness_risk import predict_wellness_risk
from services.wellness_suggestions import generate_wellness_suggestions, SUGGESTION_DISCLAIMER
from services.support_resources import (
    get_support_resources,
    get_support_resources_by_category,
    EMERGENCY_CONTACTS,
    RESPONSIBLE_AI_SUPPORT_DISCLAIMER
)
from time_utils import (
    get_utc_now,
    get_local_now,
    get_local_today,
    utc_to_local,
    get_local_day_utc_range,
    format_local_datetime,
    format_local_date,
    APP_TIMEZONE
)

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('thrivespace')

app = Flask(__name__)

# Security & App Configuration
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if 'unittest' in sys.modules or os.environ.get('TESTING') == 'True' or os.environ.get('FLASK_ENV') == 'testing':
        secret_key = 'test_dev_secret_key_testing_only'
    else:
        raise RuntimeError(
            "SECRET_KEY is not configured in the environment. "
            "Please copy .env.example to .env and configure a secure SECRET_KEY before running ThriveSpace."
        )

app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Load AI / Emotion Classification Pipeline
emotion_classifier = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    top_k=1
)
logger.info('Emotion model loaded successfully!')

# Mood System Constants
MOOD_LABELS = {
    1: {'label': 'Very Low', 'emoji': '😞', 'color': '#ef4444'},
    2: {'label': 'Low', 'emoji': '🙁', 'color': '#f97316'},
    3: {'label': 'Okay', 'emoji': '😐', 'color': '#eab308'},
    4: {'label': 'Good', 'emoji': '🙂', 'color': '#22c55e'},
    5: {'label': 'Great', 'emoji': '😄', 'color': '#8b5cf6'},
}

MOOD_TAGS = [
    'Work / College',
    'Relationships',
    'Sleep',
    'Health',
    'Finance',
    'Social',
    'Personal',
    'Other'
]

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    journals = db.relationship('Journal', backref='author', lazy=True, cascade='all, delete-orphan')
    mood_entries = db.relationship('MoodEntry', backref='user', lazy=True, cascade='all, delete-orphan')
    burnout_assessments = db.relationship('BurnoutAssessment', backref='user', lazy=True, cascade='all, delete-orphan')
    goals = db.relationship('Goal', backref='user', lazy=True, cascade='all, delete-orphan')
    habits = db.relationship('Habit', backref='user', lazy=True, cascade='all, delete-orphan')
    habit_logs = db.relationship('HabitLog', backref='user', lazy=True, cascade='all, delete-orphan')

class Journal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sentiment = db.Column(db.String(50), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    emotion = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class MoodEntry(db.Model):
    __tablename__ = 'mood_entry'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    mood_rating = db.Column(db.Integer, nullable=False)
    mood_tag = db.Column(db.String(50), nullable=True)
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    @property
    def label(self):
        return MOOD_LABELS.get(self.mood_rating, {}).get('label', 'Unknown')

    @property
    def emoji(self):
        return MOOD_LABELS.get(self.mood_rating, {}).get('emoji', '😐')

    @property
    def color(self):
        return MOOD_LABELS.get(self.mood_rating, {}).get('color', '#8b5cf6')

    @property
    def local_created_at(self):
        return utc_to_local(self.created_at)

    @property
    def local_date(self):
        return self.local_created_at.date()

class BurnoutAssessment(db.Model):
    __tablename__ = 'burnout_assessment'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    
    # Self-reported lifestyle indicators
    sleep_hours = db.Column(db.Float, nullable=False)
    work_study_hours = db.Column(db.Float, nullable=False)
    stress_level = db.Column(db.Float, nullable=False)
    physical_activity_minutes = db.Column(db.Float, nullable=False)
    
    # Derived or training-data-imputed signals
    mood_rating = db.Column(db.Float, nullable=True)
    journal_sentiment = db.Column(db.Float, nullable=True)
    negative_emotion_frequency = db.Column(db.Float, nullable=True)
    
    # Model predictions
    risk_score = db.Column(db.Integer, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)  # 'Low', 'Moderate', 'Elevated'
    risk_class = db.Column(db.Integer, nullable=False)      # 0, 1, 2
    
    # Serialized factors & distributions
    contributing_factors_json = db.Column(db.Text, nullable=False, default='[]')
    probabilities_json = db.Column(db.Text, nullable=False, default='{}')
    
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    @property
    def local_created_at(self):
        return utc_to_local(self.created_at)

    @property
    def local_date(self):
        return self.local_created_at.date()

    @property
    def contributing_factors(self):
        try:
            return json.loads(self.contributing_factors_json)
        except Exception:
            return []

    @property
    def probabilities(self):
        try:
            return json.loads(self.probabilities_json)
        except Exception:
            return {'low': 0.0, 'moderate': 0.0, 'elevated': 0.0}

    @property
    def missing_features(self):
        missing = []
        if self.mood_rating is None:
            missing.append('mood_rating')
        if self.journal_sentiment is None:
            missing.append('journal_sentiment')
        if self.negative_emotion_frequency is None:
            missing.append('negative_emotion_frequency')
        return missing

    @property
    def badge_color(self):
        if self.risk_level == 'Low':
            return '#10b981'
        elif self.risk_level == 'Moderate':
            return '#f59e0b'
        return '#ef4444'

    @property
    def badge_bg(self):
        if self.risk_level == 'Low':
            return '#ecfdf5'
        elif self.risk_level == 'Moderate':
            return '#fffbeb'
        return '#fef2f2'

    @property
    def data_sources(self):
        sources = [
            {'feature': 'Sleep', 'status': 'available', 'description': 'User provided'},
            {'feature': 'Work/Study', 'status': 'available', 'description': 'User provided'},
            {'feature': 'Stress', 'status': 'available', 'description': 'User provided'},
            {'feature': 'Physical Activity', 'status': 'available', 'description': 'User provided'},
        ]
        if self.mood_rating is not None:
            sources.append({
                'feature': 'Mood',
                'status': 'available',
                'description': f'{self.mood_rating:.1f}/5 (Recent check-in average)'
            })
        else:
            sources.append({
                'feature': 'Mood',
                'status': 'missing',
                'description': 'No mood check-ins available'
            })

        if self.journal_sentiment is not None:
            sources.append({
                'feature': 'Journal Sentiment',
                'status': 'available',
                'description': 'Recent journals'
            })
        else:
            sources.append({
                'feature': 'Journal Sentiment',
                'status': 'missing',
                'description': 'No journals available'
            })

        if self.negative_emotion_frequency is not None:
            sources.append({
                'feature': 'Negative Emotion Frequency',
                'status': 'available',
                'description': 'Recent journals'
            })
        else:
            sources.append({
                'feature': 'Negative Emotion Frequency',
                'status': 'missing',
                'description': 'No journals available'
            })

        return sources

VALID_GOAL_CATEGORIES = [
    'Sleep',
    'Physical Activity',
    'Journaling',
    'Mood',
    'Study / Work',
    'Self-Care',
    'Other'
]

VALID_HABIT_CATEGORIES = [
    'Sleep',
    'Exercise',
    'Journaling',
    'Mood Check-in',
    'Mindfulness',
    'Self-Care',
    'Other'
]

VALID_HABIT_FREQUENCIES = ['Daily', 'Weekly']

class Goal(db.Model):
    __tablename__ = 'goal'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=False, default='Self-Care')
    target_value = db.Column(db.Float, nullable=False, default=1.0)
    current_value = db.Column(db.Float, nullable=False, default=0.0)
    unit = db.Column(db.String(50), nullable=False, default='times')
    deadline = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(20), nullable=False, default='active')  # 'active', 'completed'
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    @property
    def progress_percentage(self):
        if self.target_value <= 0:
            return 100.0 if self.status == 'completed' else 0.0
        pct = (self.current_value / self.target_value) * 100.0
        return round(min(100.0, max(0.0, pct)), 1)

    @property
    def is_completed(self):
        return self.status == 'completed' or (self.target_value > 0 and self.current_value >= self.target_value)


class Habit(db.Model):
    __tablename__ = 'habit'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='Self-Care')
    frequency = db.Column(db.String(20), nullable=False, default='Daily')  # 'Daily', 'Weekly'
    target_count = db.Column(db.Integer, nullable=False, default=1)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    logs = db.relationship('HabitLog', backref='habit', lazy=True, cascade='all, delete-orphan')


class HabitLog(db.Model):
    __tablename__ = 'habit_log'
    id = db.Column(db.Integer, primary_key=True)
    habit_id = db.Column(db.Integer, db.ForeignKey('habit.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    completed_date = db.Column(db.Date, nullable=False)
    completed = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.UniqueConstraint('habit_id', 'completed_date', name='uq_habit_date'),
    )


def get_user_journal_signals(user_id, limit=5):
    journals = Journal.query.filter_by(user_id=user_id).order_by(Journal.created_at.desc()).limit(limit).all()
    if not journals:
        return {'count': 0, 'avg_sentiment': None, 'neg_emotion_ratio': None}
    
    sentiments = [j.sentiment_score for j in journals if j.sentiment_score is not None]
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else None
    
    neg_emotions = {'sadness', 'fear', 'anger', 'disgust'}
    neg_count = sum(1 for j in journals if j.emotion and j.emotion.lower() in neg_emotions)
    neg_ratio = neg_count / len(journals)
    
    return {
        'count': len(journals),
        'avg_sentiment': round(avg_sent, 2) if avg_sent is not None else None,
        'neg_emotion_ratio': round(neg_ratio, 2)
    }


def get_habit_stats_for_user(user_id):
    today = get_local_today()
    start_of_week = today - timedelta(days=today.weekday())  # Monday of current local week
    
    habits = Habit.query.filter_by(user_id=user_id).order_by(Habit.created_at.desc()).all()
    habit_items = []
    today_completed_count = 0
    
    for h in habits:
        today_log = HabitLog.query.filter_by(habit_id=h.id, completed_date=today, completed=True).first()
        is_completed_today = today_log is not None
        if is_completed_today:
            today_completed_count += 1
            
        week_count = HabitLog.query.filter(
            HabitLog.habit_id == h.id,
            HabitLog.completed == True,
            HabitLog.completed_date >= start_of_week,
            HabitLog.completed_date <= today
        ).count()
        
        habit_items.append({
            'id': h.id,
            'name': h.name,
            'category': h.category,
            'frequency': h.frequency,
            'target_count': h.target_count,
            'is_completed_today': is_completed_today,
            'week_count': week_count,
            'created_at': h.created_at
        })
        
    return habit_items, today_completed_count

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Template Filters for Timezone Formatting
@app.template_filter('format_datetime')
def jinja_format_datetime(dt):
    return format_local_datetime(dt)

@app.template_filter('format_date')
def jinja_format_date(dt):
    return format_local_date(dt)

# Safe NLP Extraction Helper
def analyze_journal_text(content):
    try:
        result = emotion_classifier(content, truncation=True, max_length=512)
        emotion = result[0][0]['label']
    except Exception as e:
        logger.warning('Emotion classification warning: %s', e)
        emotion = 'neutral'

    try:
        blob = TextBlob(content)
        score = round(float(blob.sentiment.polarity), 3)
        if score > 0.05:
            sentiment = 'Positive'
        elif score < -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    except Exception as e:
        logger.warning('Sentiment analysis warning: %s', e)
        score = 0.0
        sentiment = 'Neutral'

    return emotion, sentiment, score

# Mood System Helpers
def get_today_mood_entry(user_id):
    today_local = get_local_today()
    start_utc, end_utc = get_local_day_utc_range(today_local)
    start_naive = start_utc.replace(tzinfo=None)
    end_naive = end_utc.replace(tzinfo=None)

    return MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.created_at >= start_naive,
        MoodEntry.created_at <= end_naive
    ).order_by(MoodEntry.created_at.desc()).first()

def get_user_mood_summary(user_id):
    today_local = get_local_today()
    seven_days_ago = today_local - timedelta(days=6)
    prior_seven_start = today_local - timedelta(days=13)
    prior_seven_end = today_local - timedelta(days=7)

    start_7d_utc, _ = get_local_day_utc_range(seven_days_ago)
    _, end_today_utc = get_local_day_utc_range(today_local)
    start_prior_utc, _ = get_local_day_utc_range(prior_seven_start)
    _, end_prior_utc = get_local_day_utc_range(prior_seven_end)

    start_7d_naive = start_7d_utc.replace(tzinfo=None)
    end_today_naive = end_today_utc.replace(tzinfo=None)
    start_prior_naive = start_prior_utc.replace(tzinfo=None)
    end_prior_naive = end_prior_utc.replace(tzinfo=None)

    # 7-day entries
    recent_entries = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.created_at >= start_7d_naive,
        MoodEntry.created_at <= end_today_naive
    ).order_by(MoodEntry.created_at.asc()).all()

    daily_map = {}
    for entry in recent_entries:
        daily_map[entry.local_date] = entry

    recent_ratings = [entry.mood_rating for entry in daily_map.values()]
    avg_7d = round(sum(recent_ratings) / len(recent_ratings), 1) if recent_ratings else None

    # Prior 7-day entries
    prior_entries = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.created_at >= start_prior_naive,
        MoodEntry.created_at <= end_prior_naive
    ).order_by(MoodEntry.created_at.asc()).all()

    prior_map = {}
    for entry in prior_entries:
        prior_map[entry.local_date] = entry
    prior_ratings = [entry.mood_rating for entry in prior_map.values()]
    avg_prior = round(sum(prior_ratings) / len(prior_ratings), 1) if prior_ratings else None

    # Trend message
    if avg_7d is not None and avg_prior is not None:
        diff = avg_7d - avg_prior
        if diff >= 0.3:
            trend_msg = 'Your average mood has improved over the last 7 days.'
        elif diff <= -0.3:
            trend_msg = 'Your average mood has declined slightly compared to last week.'
        else:
            trend_msg = 'Your average mood has remained steady over the last 7 days.'
    elif len(recent_ratings) >= 2:
        half = len(recent_ratings) // 2
        first_half_avg = sum(recent_ratings[:half]) / half
        second_half_avg = sum(recent_ratings[half:]) / (len(recent_ratings) - half)
        diff = second_half_avg - first_half_avg
        if diff >= 0.3:
            trend_msg = 'Your mood shows an upward trend over recent check-ins.'
        elif diff <= -0.3:
            trend_msg = 'Your mood has dipped slightly over recent check-ins.'
        else:
            trend_msg = 'Your mood has remained relatively consistent over recent check-ins.'
    else:
        trend_msg = 'Check in for a few more days to see your mood trend.'

    # Chart series
    chart_labels = []
    chart_values = []
    for i in range(7):
        d = seven_days_ago + timedelta(days=i)
        chart_labels.append(d.strftime('%b %d'))
        chart_values.append(daily_map[d].mood_rating if d in daily_map else None)

    latest_entry = MoodEntry.query.filter_by(user_id=user_id).order_by(MoodEntry.created_at.desc()).first()

    return {
        'latest_entry': latest_entry,
        'avg_7d': avg_7d,
        'trend_msg': trend_msg,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
        'has_enough_chart_data': len(recent_ratings) >= 2
    }

def get_user_wellness_signals(user_id):
    """
    Derives real personal signals from user's actual history:
      - mood_rating: Average of recent MoodEntry ratings (up to 14 entries), or None if no entries exist
      - journal_sentiment: Average sentiment_score of recent journals (up to 10 entries), or None if no journals
      - negative_emotion_frequency: Proportion of recent journals with negative emotion ('sadness', 'anger', 'fear', 'disgust'), or None if no journals
    """
    mood_entries = MoodEntry.query.filter_by(user_id=user_id).order_by(MoodEntry.created_at.desc()).limit(14).all()
    if mood_entries:
        derived_mood = round(sum(m.mood_rating for m in mood_entries) / len(mood_entries), 2)
        mood_sample_count = len(mood_entries)
    else:
        derived_mood = None
        mood_sample_count = 0

    recent_journals = Journal.query.filter_by(user_id=user_id).order_by(Journal.created_at.desc()).limit(10).all()
    if recent_journals:
        valid_scores = [j.sentiment_score for j in recent_journals if j.sentiment_score is not None]
        derived_sentiment = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None

        neg_emotions = {'sadness', 'anger', 'fear', 'disgust'}
        neg_count = sum(1 for j in recent_journals if (j.emotion or '').lower() in neg_emotions)
        derived_neg_freq = round(neg_count / len(recent_journals), 3)
        journal_sample_count = len(recent_journals)
    else:
        derived_sentiment = None
        derived_neg_freq = None
        journal_sample_count = 0

    return {
        'mood_rating': derived_mood,
        'mood_sample_count': mood_sample_count,
        'journal_sentiment': derived_sentiment,
        'negative_emotion_frequency': derived_neg_freq,
        'journal_sample_count': journal_sample_count,
        'has_mood_data': derived_mood is not None,
        'has_journal_data': derived_sentiment is not None
    }

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect('/journals')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect('/journals')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email is already registered. Please login.', 'danger')
            return redirect('/register')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully! Please sign in.', 'success')
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/journals')

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect('/journals')
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out securely.', 'info')
    return redirect('/login')

@app.route('/mood', methods=['GET', 'POST'])
@login_required
def mood_checkin():
    today_entry = get_today_mood_entry(current_user.id)

    if request.method == 'POST':
        rating_raw = request.form.get('mood_rating')
        tag = request.form.get('mood_tag', '').strip()
        note = request.form.get('note', '').strip()

        try:
            rating = int(rating_raw)
            if rating < 1 or rating > 5:
                raise ValueError()
        except (TypeError, ValueError):
            flash('Please select a valid mood rating between 1 and 5.', 'danger')
            summary = get_user_mood_summary(current_user.id)
            return render_template(
                'mood_checkin.html',
                today_entry=today_entry,
                mood_labels=MOOD_LABELS,
                mood_tags=MOOD_TAGS,
                summary=summary
            )

        if tag and tag not in MOOD_TAGS:
            tag = 'Other'

        now_utc = get_utc_now().replace(tzinfo=None)

        if today_entry:
            today_entry.mood_rating = rating
            today_entry.mood_tag = tag if tag else None
            today_entry.note = note if note else None
            today_entry.created_at = now_utc
            db.session.commit()
            flash('Today\'s mood check-in has been updated!', 'success')
        else:
            new_entry = MoodEntry(
                user_id=current_user.id,
                mood_rating=rating,
                mood_tag=tag if tag else None,
                note=note if note else None,
                created_at=now_utc
            )
            db.session.add(new_entry)
            db.session.commit()
            flash('Your daily mood has been recorded!', 'success')

        return redirect('/journals')

    summary = get_user_mood_summary(current_user.id)
    return render_template(
        'mood_checkin.html',
        today_entry=today_entry,
        mood_labels=MOOD_LABELS,
        mood_tags=MOOD_TAGS,
        summary=summary
    )

@app.route('/mood/history')
@login_required
def mood_history():
    entries = MoodEntry.query.filter_by(user_id=current_user.id).order_by(MoodEntry.created_at.desc()).all()
    summary = get_user_mood_summary(current_user.id)
    return render_template('mood_history.html', entries=entries, summary=summary)

@app.route('/journals')
@login_required
def journals():
    all_journals = Journal.query.filter_by(user_id=current_user.id).order_by(Journal.created_at.desc()).all()
    today_mood = get_today_mood_entry(current_user.id)
    mood_summary = get_user_mood_summary(current_user.id)
    latest_journal = all_journals[0] if all_journals else None
    latest_assessment = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).first()

    # Phase 4 Dashboard: Goals, Habits, Suggestions
    active_goals = Goal.query.filter_by(user_id=current_user.id, status='active').order_by(Goal.created_at.desc()).limit(3).all()
    completed_goals_count = Goal.query.filter_by(user_id=current_user.id, status='completed').count()
    habit_items, today_habits_completed = get_habit_stats_for_user(current_user.id)
    journal_signals = get_user_journal_signals(current_user.id)

    suggestions = generate_wellness_suggestions(
        latest_assessment=latest_assessment,
        recent_mood_summary=mood_summary,
        recent_journal_signals=journal_signals,
        active_goals=active_goals,
        completed_goals_count=completed_goals_count,
        today_habits_completed=today_habits_completed,
        active_habits_count=len(habit_items)
    )

    return render_template(
        'journals.html',
        journals=all_journals,
        today_mood=today_mood,
        mood_summary=mood_summary,
        latest_journal=latest_journal,
        latest_assessment=latest_assessment,
        suggestions=suggestions,
        active_goals=active_goals,
        habits=habit_items[:4],
        today_habits_completed=today_habits_completed,
        disclaimer=SUGGESTION_DISCLAIMER
    )

@app.route('/create-journal', methods=['GET', 'POST'])
@login_required
def create_journal():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()

        if not title or not content:
            flash('Title and content cannot be empty.', 'danger')
            return render_template('create_journal.html')

        emotion, sentiment, score = analyze_journal_text(content)

        journal = Journal(
            title=title,
            content=content,
            user_id=current_user.id,
            sentiment=sentiment,
            sentiment_score=score,
            emotion=emotion,
            created_at=get_utc_now().replace(tzinfo=None),
            updated_at=get_utc_now().replace(tzinfo=None)
        )
        db.session.add(journal)
        db.session.commit()

        flash('Journal entry saved and analyzed.', 'success')
        return redirect('/journals')

    return render_template('create_journal.html')

@app.route('/delete-journal/<int:id>', methods=['POST'])
@login_required
def delete_journal(id):
    journal = db.session.get(Journal, id)
    if not journal:
        flash('Journal not found.', 'danger')
        return redirect('/journals')

    if journal.user_id == current_user.id:
        db.session.delete(journal)
        db.session.commit()
        flash('Journal entry deleted.', 'info')
    else:
        flash('Unauthorized action.', 'danger')
    return redirect('/journals')

@app.route('/update-journal/<int:id>', methods=['GET', 'POST'])
@login_required
def update_journal(id):
    journal = db.session.get(Journal, id)
    if not journal:
        flash('Journal not found.', 'danger')
        return redirect('/journals')

    if journal.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect('/journals')

    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()

        if not title or not content:
            flash('Title and content cannot be empty.', 'danger')
            return render_template('update_journal.html', journal=journal)

        emotion, sentiment, score = analyze_journal_text(content)

        journal.title = title
        journal.content = content
        journal.emotion = emotion
        journal.sentiment = sentiment
        journal.sentiment_score = score
        journal.updated_at = get_utc_now().replace(tzinfo=None)

        db.session.commit()
        flash('Journal entry updated successfully.', 'success')
        return redirect('/journals')

    return render_template('update_journal.html', journal=journal)

@app.route('/support')
@app.route('/therapists')
def therapists():
    resources = get_support_resources()
    resources_by_cat = get_support_resources_by_category()

    support_cta = None
    if current_user.is_authenticated:
        latest_assessment = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).first()
        is_elevated = False
        if latest_assessment is not None:
            level = getattr(latest_assessment, 'risk_level', '')
            r_class = getattr(latest_assessment, 'risk_class', None)
            if level in ['Moderate', 'Elevated'] or r_class in [1, 2]:
                is_elevated = True

        if is_elevated:
            support_cta = "Consider speaking with a qualified professional if these concerns are persistent or affecting your daily life."
        else:
            support_cta = "Professional support is available if you would like help understanding concerns that are affecting your wellbeing."

    return render_template(
        'therapists.html',
        resources=resources,
        resources_by_category=resources_by_cat,
        emergency_contacts=EMERGENCY_CONTACTS,
        disclaimer=RESPONSIBLE_AI_SUPPORT_DISCLAIMER,
        support_cta=support_cta
    )

@app.route('/get-therapists')
def get_therapists():
    # 100% free, local curated directory response without external API dependencies
    resources = get_support_resources()
    return jsonify({
        'status': 'success',
        'source': 'local_verified_directory',
        'results': [
            {
                'name': r['name'],
                'category': r['category'],
                'phone': r['phone'],
                'vicinity': r['region'],
                'website': r['website'],
                'description': r['description'],
                'rating': 'Verified'
            }
            for r in resources
        ]
    }), 200

@app.route('/burnout', methods=['GET', 'POST'])
@login_required
def burnout_assessment():
    signals = get_user_wellness_signals(current_user.id)
    latest_assessment = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).first()
    history = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).limit(10).all()

    if request.method == 'POST':
        sleep_raw = request.form.get('sleep_hours', '').strip()
        work_raw = request.form.get('work_study_hours', '').strip()
        stress_raw = request.form.get('stress_level', '').strip()
        activity_raw = request.form.get('physical_activity_minutes', '').strip()

        errors = []
        sleep_hours = None
        work_study_hours = None
        stress_level = None
        physical_activity_minutes = None

        try:
            sleep_hours = float(sleep_raw)
            if not (0.0 <= sleep_hours <= 24.0):
                errors.append('Sleep hours must be between 0 and 24.')
        except (ValueError, TypeError):
            errors.append('Please provide a valid number for sleep hours.')

        try:
            work_study_hours = float(work_raw)
            if not (0.0 <= work_study_hours <= 24.0):
                errors.append('Work/study hours must be between 0 and 24.')
        except (ValueError, TypeError):
            errors.append('Please provide a valid number for work/study hours.')

        try:
            stress_level = float(stress_raw)
            if not (1.0 <= stress_level <= 10.0):
                errors.append('Stress level must be between 1 and 10.')
        except (ValueError, TypeError):
            errors.append('Please provide a valid stress level (1 to 10).')

        try:
            physical_activity_minutes = float(activity_raw)
            if not (0.0 <= physical_activity_minutes <= 1440.0):
                errors.append('Physical activity must be between 0 and 1440 minutes.')
        except (ValueError, TypeError):
            errors.append('Please provide a valid number for physical activity minutes.')

        if errors:
            for err in errors:
                flash(err, 'danger')
            return render_template(
                'burnout_assessment.html',
                signals=signals,
                latest_assessment=latest_assessment,
                history=history,
                form_data=request.form
            )

        feature_dict = {
            'sleep_hours': sleep_hours,
            'work_study_hours': work_study_hours,
            'stress_level': stress_level,
            'physical_activity_minutes': physical_activity_minutes,
            'mood_rating': signals['mood_rating'],
            'journal_sentiment': signals['journal_sentiment'],
            'negative_emotion_frequency': signals['negative_emotion_frequency']
        }

        try:
            prediction = predict_wellness_risk(feature_dict)
        except Exception as e:
            flash(f'Assessment calculation error: {str(e)}', 'danger')
            return render_template(
                'burnout_assessment.html',
                signals=signals,
                latest_assessment=latest_assessment,
                history=history,
                form_data=request.form
            )

        now_utc = get_utc_now().replace(tzinfo=None)
        new_assessment = BurnoutAssessment(
            user_id=current_user.id,
            sleep_hours=sleep_hours,
            work_study_hours=work_study_hours,
            stress_level=stress_level,
            physical_activity_minutes=physical_activity_minutes,
            mood_rating=signals['mood_rating'],
            journal_sentiment=signals['journal_sentiment'],
            negative_emotion_frequency=signals['negative_emotion_frequency'],
            risk_score=prediction['risk_score'],
            risk_level=prediction['risk_label'],
            risk_class=prediction['risk_class'],
            probabilities_json=json.dumps(prediction['probabilities']),
            contributing_factors_json=json.dumps(prediction['contributing_factors']),
            created_at=now_utc
        )
        db.session.add(new_assessment)
        db.session.commit()

        flash('Your wellness risk assessment has been calculated.', 'success')
        return redirect('/burnout')

    return render_template(
        'burnout_assessment.html',
        signals=signals,
        latest_assessment=latest_assessment,
        history=history,
        form_data={}
    )

@app.route('/burnout/<int:assessment_id>/delete', methods=['POST'])
@login_required
def delete_assessment(assessment_id):
    assessment = BurnoutAssessment.query.get_or_404(assessment_id)
    if assessment.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect('/burnout')

    db.session.delete(assessment)
    db.session.commit()
    flash('Assessment record deleted.', 'info')
    return redirect('/burnout')

# -------------------------------------------------------------
# Phase 4: Goals & Habits Routes
# -------------------------------------------------------------
@app.route('/goals')
@login_required
def goals():
    active_goals = Goal.query.filter_by(user_id=current_user.id, status='active').order_by(Goal.created_at.desc()).all()
    completed_goals = Goal.query.filter_by(user_id=current_user.id, status='completed').order_by(Goal.updated_at.desc()).all()
    habit_items, today_habits_completed = get_habit_stats_for_user(current_user.id)
    
    latest_assessment = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).first()
    mood_summary = get_user_mood_summary(current_user.id)
    journal_signals = get_user_journal_signals(current_user.id)
    
    suggestions = generate_wellness_suggestions(
        latest_assessment=latest_assessment,
        recent_mood_summary=mood_summary,
        recent_journal_signals=journal_signals,
        active_goals=active_goals,
        completed_goals_count=len(completed_goals),
        today_habits_completed=today_habits_completed,
        active_habits_count=len(habit_items)
    )
    
    return render_template(
        'goals.html',
        active_goals=active_goals,
        completed_goals=completed_goals,
        habits=habit_items,
        today_habits_completed=today_habits_completed,
        suggestions=suggestions,
        categories=VALID_GOAL_CATEGORIES,
        habit_categories=VALID_HABIT_CATEGORIES,
        habit_frequencies=VALID_HABIT_FREQUENCIES,
        disclaimer=SUGGESTION_DISCLAIMER
    )

@app.route('/goals/create', methods=['POST'])
@login_required
def create_goal():
    title = request.form.get('title', '').strip()
    category = request.form.get('category', 'Self-Care').strip()
    target_val_str = request.form.get('target_value', '1.0').strip()
    current_val_str = request.form.get('current_value', '0.0').strip()
    unit = request.form.get('unit', 'times').strip()
    deadline_str = request.form.get('deadline', '').strip()
    description = request.form.get('description', '').strip()

    if not title or len(title) < 2:
        flash('Goal title must be at least 2 characters.', 'error')
        return redirect('/goals')

    if category not in VALID_GOAL_CATEGORIES:
        category = 'Self-Care'

    try:
        target_value = float(target_val_str)
        if target_value <= 0:
            flash('Target value must be greater than 0.', 'error')
            return redirect('/goals')
    except (ValueError, TypeError):
        flash('Invalid target value provided.', 'error')
        return redirect('/goals')

    try:
        current_value = float(current_val_str)
        if current_value < 0:
            flash('Current progress cannot be negative.', 'error')
            return redirect('/goals')
    except (ValueError, TypeError):
        current_value = 0.0

    deadline = None
    if deadline_str:
        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid deadline date format (expected YYYY-MM-DD).', 'error')
            return redirect('/goals')

    status = 'completed' if current_value >= target_value else 'active'

    new_goal = Goal(
        user_id=current_user.id,
        title=title,
        description=description if description else None,
        category=category,
        target_value=round(target_value, 2),
        current_value=round(current_value, 2),
        unit=unit if unit else 'times',
        deadline=deadline,
        status=status
    )
    db.session.add(new_goal)
    db.session.commit()

    flash(f'Goal "{title}" created successfully!', 'success')
    return redirect('/goals')

@app.route('/goals/<int:id>/edit', methods=['POST'])
@login_required
def edit_goal(id):
    goal = Goal.query.filter_by(id=id, user_id=current_user.id).first()
    if not goal:
        flash('Goal not found or unauthorized.', 'error')
        return redirect('/goals'), 404

    title = request.form.get('title', '').strip()
    target_val_str = request.form.get('target_value', '').strip()
    current_val_str = request.form.get('current_value', '').strip()
    description = request.form.get('description', '').strip()
    deadline_str = request.form.get('deadline', '').strip()

    if title and len(title) >= 2:
        goal.title = title
    if description is not None:
        goal.description = description if description else None

    if target_val_str:
        try:
            val = float(target_val_str)
            if val > 0:
                goal.target_value = round(val, 2)
            else:
                flash('Target value must be greater than 0.', 'error')
                return redirect('/goals')
        except ValueError:
            flash('Invalid target value.', 'error')
            return redirect('/goals')

    if current_val_str:
        try:
            val = float(current_val_str)
            if val >= 0:
                goal.current_value = round(val, 2)
            else:
                flash('Current progress cannot be negative.', 'error')
                return redirect('/goals')
        except ValueError:
            flash('Invalid current progress value.', 'error')
            return redirect('/goals')

    if deadline_str:
        try:
            goal.deadline = datetime.strptime(deadline_str, '%Y-%m-%d').date()
        except ValueError:
            pass

    if goal.current_value >= goal.target_value:
        goal.status = 'completed'
    else:
        goal.status = 'active'

    goal.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    flash(f'Goal "{goal.title}" updated.', 'success')
    return redirect('/goals')

@app.route('/goals/<int:id>/complete', methods=['POST'])
@login_required
def complete_goal(id):
    goal = Goal.query.filter_by(id=id, user_id=current_user.id).first()
    if not goal:
        flash('Goal not found or unauthorized.', 'error')
        return redirect('/goals'), 404

    if goal.status == 'completed':
        goal.status = 'active'
        flash(f'Goal "{goal.title}" marked active.', 'info')
    else:
        goal.status = 'completed'
        if goal.current_value < goal.target_value:
            goal.current_value = goal.target_value
        flash(f'Goal "{goal.title}" completed! Well done.', 'success')

    goal.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return redirect(request.referrer or '/goals')

@app.route('/goals/<int:id>/delete', methods=['POST'])
@login_required
def delete_goal(id):
    goal = Goal.query.filter_by(id=id, user_id=current_user.id).first()
    if not goal:
        flash('Goal not found or unauthorized.', 'error')
        return redirect('/goals'), 404

    title = goal.title
    db.session.delete(goal)
    db.session.commit()
    flash(f'Goal "{title}" deleted.', 'info')
    return redirect('/goals')

@app.route('/habits/create', methods=['POST'])
@login_required
def create_habit():
    name = request.form.get('name', '').strip()
    category = request.form.get('category', 'Self-Care').strip()
    frequency = request.form.get('frequency', 'Daily').strip()
    target_count_str = request.form.get('target_count', '1').strip()

    if not name or len(name) < 2:
        flash('Habit name must be at least 2 characters.', 'error')
        return redirect('/goals')

    if category not in VALID_HABIT_CATEGORIES:
        category = 'Self-Care'

    if frequency not in VALID_HABIT_FREQUENCIES:
        frequency = 'Daily'

    try:
        target_count = max(1, int(target_count_str))
    except (ValueError, TypeError):
        target_count = 1

    new_habit = Habit(
        user_id=current_user.id,
        name=name,
        category=category,
        frequency=frequency,
        target_count=target_count
    )
    db.session.add(new_habit)
    db.session.commit()

    flash(f'Habit "{name}" added!', 'success')
    return redirect('/goals')

@app.route('/habits/<int:id>/toggle', methods=['POST'])
@login_required
def toggle_habit(id):
    habit = Habit.query.filter_by(id=id, user_id=current_user.id).first()
    if not habit:
        flash('Habit not found or unauthorized.', 'error')
        return redirect('/goals'), 404

    today = get_local_today()
    existing_log = HabitLog.query.filter_by(habit_id=habit.id, completed_date=today).first()

    if existing_log:
        db.session.delete(existing_log)
        db.session.commit()
        flash(f'Habit "{habit.name}" marked uncompleted for today.', 'info')
    else:
        new_log = HabitLog(
            habit_id=habit.id,
            user_id=current_user.id,
            completed_date=today,
            completed=True
        )
        db.session.add(new_log)
        db.session.commit()
        flash(f'Habit "{habit.name}" completed for today!', 'success')

    return redirect(request.referrer or '/goals')

@app.route('/habits/<int:id>/delete', methods=['POST'])
@login_required
def delete_habit(id):
    habit = Habit.query.filter_by(id=id, user_id=current_user.id).first()
    if not habit:
        flash('Habit not found or unauthorized.', 'error')
        return redirect('/goals'), 404

    name = habit.name
    db.session.delete(habit)
    db.session.commit()
    flash(f'Habit "{name}" deleted.', 'info')
    return redirect('/goals')

@app.route('/profile')
@login_required
def profile():
    journal_count = Journal.query.filter_by(user_id=current_user.id).count()
    mood_count = MoodEntry.query.filter_by(user_id=current_user.id).count()
    assessment_count = BurnoutAssessment.query.filter_by(user_id=current_user.id).count()
    latest_assessment = BurnoutAssessment.query.filter_by(user_id=current_user.id).order_by(BurnoutAssessment.created_at.desc()).first()
    summary = get_user_mood_summary(current_user.id)

    # Phase 4 Profile Stats
    active_goals_count = Goal.query.filter_by(user_id=current_user.id, status='active').count()
    completed_goals_count = Goal.query.filter_by(user_id=current_user.id, status='completed').count()
    habits = Habit.query.filter_by(user_id=current_user.id).all()
    active_habits_count = len(habits)
    today = get_local_today()
    today_habits_completed = HabitLog.query.filter(
        HabitLog.user_id == current_user.id,
        HabitLog.completed_date == today,
        HabitLog.completed == True
    ).count()

    return render_template(
        'profile.html',
        journal_count=journal_count,
        mood_count=mood_count,
        assessment_count=assessment_count,
        latest_assessment=latest_assessment,
        summary=summary,
        active_goals_count=active_goals_count,
        completed_goals_count=completed_goals_count,
        active_habits_count=active_habits_count,
        today_habits_completed=today_habits_completed
    )

# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------
@app.errorhandler(400)
def bad_request_error(e):
    return render_template('error.html', error_code=400, error_title='Bad Request', error_message='The request could not be processed. Please check your inputs.'), 400

@app.errorhandler(CSRFError)
def csrf_error(e):
    return render_template('error.html', error_code=400, error_title='Session Security Check Failed', error_message='Your session token expired or is invalid. Please refresh the page and try again.'), 400

@app.errorhandler(403)
def forbidden_error(e):
    return render_template('error.html', error_code=403, error_title='Access Denied', error_message='You do not have permission to view or modify this resource.'), 403

@app.errorhandler(404)
def not_found_error(e):
    return render_template('error.html', error_code=404, error_title='Page Not Found', error_message='The page or resource you are looking for does not exist.'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error('Internal server error: %s', e)
    return render_template('error.html', error_code=500, error_title='Internal Server Error', error_message='Something went wrong on our end. Please try again shortly.'), 500

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    app.run(debug=debug_mode, use_reloader=False)
