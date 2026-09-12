import os
import json
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'wellness_risk_model.joblib')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_schema.json')

_pipeline = None
_schema = None

def get_model_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model artifact not found at {MODEL_PATH}. Please run ml/train_wellness_model.py first.')
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def get_feature_schema():
    global _schema
    if _schema is None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f'Schema artifact not found at {SCHEMA_PATH}.')
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            _schema = json.load(f)
    return _schema

def _identify_contributing_factors(cleaned_inputs, imputer_medians, missing_features=None):
    if missing_features is None:
        missing_features = []
    factors = []
    
    # Stress Level (scale 1-10)
    stress = cleaned_inputs.get('stress_level')
    if stress is not None and not np.isnan(stress):
        if stress >= 7:
            factors.append({
                'factor': 'High Perceived Stress',
                'severity': 'high',
                'description': f'Reported stress level ({stress:.1f}/10) is significantly elevated.',
                'tip': 'Incorporate short pause breaks, deep breathing exercises, or 5-minute mindfulness pauses throughout your day.'
            })
        elif stress >= 5:
            factors.append({
                'factor': 'Moderate Stress Level',
                'severity': 'medium',
                'description': f'Reported stress level ({stress:.1f}/10) indicates noticeable tension.',
                'tip': 'Set clear boundaries between work/study and personal time to prevent cumulative fatigue.'
            })

    # Sleep Hours (scale 0-24)
    sleep = cleaned_inputs.get('sleep_hours')
    if sleep is not None and not np.isnan(sleep):
        if sleep < 6.0:
            factors.append({
                'factor': 'Insufficient Sleep Duration',
                'severity': 'high',
                'description': f'Average sleep of {sleep:.1f} hours is below the recommended 7-9 hour restorative range.',
                'tip': 'Prioritize a consistent wind-down routine 45 minutes before bed and limit blue-light screen exposure.'
            })
        elif sleep < 7.0:
            factors.append({
                'factor': 'Slight Sleep Deficit',
                'severity': 'low',
                'description': f'Average sleep of {sleep:.1f} hours is slightly below the restorative benchmark.',
                'tip': 'Aim for an extra 30-45 minutes of sleep tonight to support cognitive recovery.'
            })

    # Work / Study Hours (scale 0-24)
    work = cleaned_inputs.get('work_study_hours')
    if work is not None and not np.isnan(work):
        if work > 9.0:
            factors.append({
                'factor': 'Extended Work / Study Demands',
                'severity': 'high',
                'description': f'Work/study duration of {work:.1f} hours per day indicates sustained cognitive load.',
                'tip': 'Implement the Pomodoro technique (25 min work, 5 min break) and ensure clear stopping times.'
            })
        elif work > 7.5:
            factors.append({
                'factor': 'Demanding Schedule',
                'severity': 'medium',
                'description': f'Daily commitments of {work:.1f} hours represent a full workload.',
                'tip': 'Protect your evenings for recovery, hobbies, and low-stimulation rest.'
            })

    # Physical Activity (minutes)
    activity = cleaned_inputs.get('physical_activity_minutes')
    if activity is not None and not np.isnan(activity):
        if activity < 20:
            factors.append({
                'factor': 'Limited Physical Movement',
                'severity': 'medium',
                'description': f'Daily movement ({int(activity)} minutes) is low.',
                'tip': 'A brisk 15-20 minute walk outside can boost dopamine and reduce cortisol levels.'
            })

    # Negative Emotion Frequency in Journals (0.0 - 1.0)
    neg_freq = cleaned_inputs.get('negative_emotion_frequency')
    if neg_freq is not None and not np.isnan(neg_freq):
        if neg_freq > 0.40:
            factors.append({
                'factor': 'Frequent Negative Emotional Themes',
                'severity': 'high',
                'description': f'{int(neg_freq*100)}% of recent journal emotional expressions reflect sadness, fear, anger, or strain.',
                'tip': 'Try expressive writing: write freely about what is burdening you, then identify one small aspect within your control.'
            })
        elif neg_freq > 0.25:
            factors.append({
                'factor': 'Emerging Emotional Strain',
                'severity': 'medium',
                'description': f'Journal reflections show moderate recurrence ({int(neg_freq*100)}%) of distressing emotions.',
                'tip': 'Acknowledge these feelings without self-judgment; journaling regularly can help process them.'
            })

    # Mood Rating (scale 1-5)
    mood = cleaned_inputs.get('mood_rating')
    if mood is not None and not np.isnan(mood):
        if mood <= 2.0:
            factors.append({
                'factor': 'Low Mood Trend',
                'severity': 'high',
                'description': f'Recent mood check-ins average {mood:.1f} out of 5.',
                'tip': 'Reach out to a trusted friend, family member, or professional to talk through how you are feeling.'
            })
        elif mood <= 3.0:
            factors.append({
                'factor': 'Neutral / Subdued Mood',
                'severity': 'low',
                'description': f'Recent mood check-ins average {mood:.1f} out of 5.',
                'tip': 'Plan one small comforting activity today that brings genuine ease or joy.'
            })

    # Journal Sentiment (-1.0 to +1.0)
    sentiment = cleaned_inputs.get('journal_sentiment')
    if sentiment is not None and not np.isnan(sentiment):
        if sentiment < -0.15:
            factors.append({
                'factor': 'Negative Tone in Reflections',
                'severity': 'medium',
                'description': f'Journal sentiment polarity is negative ({sentiment:.2f}).',
                'tip': 'Consider noting three small things you are grateful for or that went reasonably well at the end of each day.'
            })

    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    factors.sort(key=lambda x: severity_order.get(x['severity'], 3))
    
    if not factors:
        if missing_features and len(missing_features) > 0:
            factors.append({
                'factor': 'No Major Strain in Available Inputs',
                'severity': 'low',
                'description': 'No significant strain signals were identified from your available self-reported inputs (sleep, work, stress, and physical activity). Missing personal signals were not assumed to be positive or negative.',
                'tip': 'Continue your daily routines, and consider logging daily moods and journal reflections to provide complete personal data.'
            })
        else:
            factors.append({
                'factor': 'No Major Contributing Signal Identified',
                'severity': 'low',
                'description': 'No major contributing strain signals were identified from your provided lifestyle and wellness indicators.',
                'tip': 'Continue maintaining consistent rest, physical activity, and balanced daily routines.'
            })
        
    return factors

def predict_wellness_risk(feature_dict):
    pipeline = get_model_pipeline()
    schema = get_feature_schema()
    
    feature_names = schema['feature_names']
    imputer_medians = schema['imputer_medians']
    
    cleaned = {}
    missing_features = []
    
    for feat in feature_names:
        val = feature_dict.get(feat)
        if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
            cleaned[feat] = np.nan
            missing_features.append(feat)
        else:
            try:
                cleaned[feat] = float(val)
            except (ValueError, TypeError):
                cleaned[feat] = np.nan
                missing_features.append(feat)
                
    if not np.isnan(cleaned['sleep_hours']):
        cleaned['sleep_hours'] = max(0.0, min(24.0, cleaned['sleep_hours']))
    if not np.isnan(cleaned['work_study_hours']):
        cleaned['work_study_hours'] = max(0.0, min(24.0, cleaned['work_study_hours']))
    if not np.isnan(cleaned['stress_level']):
        cleaned['stress_level'] = max(1.0, min(10.0, cleaned['stress_level']))
    if not np.isnan(cleaned['physical_activity_minutes']):
        cleaned['physical_activity_minutes'] = max(0.0, min(1440.0, cleaned['physical_activity_minutes']))
    if not np.isnan(cleaned['mood_rating']):
        cleaned['mood_rating'] = max(1.0, min(5.0, cleaned['mood_rating']))
    if not np.isnan(cleaned['journal_sentiment']):
        cleaned['journal_sentiment'] = max(-1.0, min(1.0, cleaned['journal_sentiment']))
    if not np.isnan(cleaned['negative_emotion_frequency']):
        cleaned['negative_emotion_frequency'] = max(0.0, min(1.0, cleaned['negative_emotion_frequency']))
        
    df_input = pd.DataFrame([[cleaned[f] for f in feature_names]], columns=feature_names)
    
    probs = pipeline.predict_proba(df_input)[0]
    pred_class = int(pipeline.predict(df_input)[0])
    
    p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])
    
    # Deterministic continuous risk score 0 - 100
    risk_score = int(round(min(100.0, max(0.0, 100.0 * (0.0 * p0 + 0.5 * p1 + 1.0 * p2)))))
    
    if risk_score <= 33:
        risk_label = 'Low'
    elif risk_score <= 66:
        risk_label = 'Moderate'
    else:
        risk_label = 'Elevated'
        
    contributing_factors = _identify_contributing_factors(cleaned, imputer_medians, missing_features=missing_features)
    
    # Transparent source breakdown
    data_sources = [
        {'feature': 'Sleep', 'status': 'available', 'description': 'User provided'},
        {'feature': 'Work/Study', 'status': 'available', 'description': 'User provided'},
        {'feature': 'Stress', 'status': 'available', 'description': 'User provided'},
        {'feature': 'Physical Activity', 'status': 'available', 'description': 'User provided'},
    ]
    if 'mood_rating' in missing_features:
        data_sources.append({'feature': 'Mood', 'status': 'missing', 'description': 'No mood check-ins available'})
    else:
        data_sources.append({'feature': 'Mood', 'status': 'available', 'description': 'Recent check-in average'})

    if 'journal_sentiment' in missing_features:
        data_sources.append({'feature': 'Journal Sentiment', 'status': 'missing', 'description': 'No journals available'})
    else:
        data_sources.append({'feature': 'Journal Sentiment', 'status': 'available', 'description': 'Recent journals'})

    if 'negative_emotion_frequency' in missing_features:
        data_sources.append({'feature': 'Negative Emotion Frequency', 'status': 'missing', 'description': 'No journals available'})
    else:
        data_sources.append({'feature': 'Negative Emotion Frequency', 'status': 'available', 'description': 'Recent journals'})

    return {
        'risk_class': pred_class,
        'risk_label': risk_label,
        'risk_score': risk_score,
        'probabilities': {
            'low': round(p0, 4),
            'moderate': round(p1, 4),
            'elevated': round(p2, 4)
        },
        'contributing_factors': contributing_factors,
        'missing_features': missing_features,
        'data_sources': data_sources,
        'cleaned_inputs': {k: (None if np.isnan(v) else round(v, 2)) for k, v in cleaned.items()},
        'is_diagnostic': False,
        'disclaimer': 'This wellness assessment is an AI-assisted self-reflection tool, not a clinical diagnosis or medical device. The underlying Random Forest model is a technical prototype evaluated on synthetic demonstration data.'
    }
