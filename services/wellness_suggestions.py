"""
ThriveSpace — Explainable Rule-Based Wellness Suggestions Service
Phase 4: Goals + Habits + Personalized Wellness Suggestions

CRITICAL RULES:
- 100% deterministic, rule-based, local open-source Python.
- Never invent missing personal data (if mood or journal data is missing, skip rules dependent on them).
- Maximum 3 suggestions returned.
- Prioritize: 1) Strongest current strain signal, 2) Second relevant wellness signal, 3) Goal/habit consistency.
- Explainable output: every suggestion contains 'suggestion', 'reason', and 'source_signal'.
- Non-clinical disclaimer attached.
"""

SUGGESTION_DISCLAIMER = (
    "Suggestions are general wellness ideas generated from your available self-reflection data. "
    "They are not medical advice or a diagnosis."
)

def generate_wellness_suggestions(
    latest_assessment=None,
    recent_mood_summary=None,
    recent_journal_signals=None,
    active_goals=None,
    completed_goals_count=0,
    today_habits_completed=0,
    active_habits_count=0
):
    """
    Generate up to 3 explainable wellness suggestions based strictly on available user data.
    """
    candidates = []
    
    # -------------------------------------------------------------
    # 1. ASSESSMENT SIGNALS (Only if latest_assessment exists)
    # -------------------------------------------------------------
    if latest_assessment is not None:
        # Perceived Stress (Scale 1-10)
        stress = getattr(latest_assessment, 'stress_level', None)
        if stress is not None and stress >= 7.0:
            candidates.append({
                'priority': 1,
                'weight': stress,
                'suggestion': "A small step could be taking a short pause away from your current task and returning with a clearer plan.",
                'reason': f"Your reported stress level ({stress:.1f}/10) is elevated.",
                'source_signal': "Perceived Stress"
            })

        # Sleep Hours
        sleep = getattr(latest_assessment, 'sleep_hours', None)
        if sleep is not None and sleep < 6.0:
            candidates.append({
                'priority': 1,
                'weight': 6.0 - sleep,
                'suggestion': "Consider protecting a consistent wind-down routine and giving yourself more time for rest.",
                'reason': f"Your reported sleep ({sleep:.1f} hrs) is below the recommended restorative range.",
                'source_signal': "Sleep Duration"
            })

        # Work / Study Hours
        work = getattr(latest_assessment, 'work_study_hours', None)
        if work is not None and work >= 9.0:
            candidates.append({
                'priority': 1,
                'weight': work - 8.0,
                'suggestion': "Consider adding short recovery breaks between focused work or study sessions.",
                'reason': f"Your reported work/study demand ({work:.1f} hrs) is relatively high.",
                'source_signal': "Work/Study Hours"
            })

        # Physical Activity
        activity = getattr(latest_assessment, 'physical_activity_minutes', None)
        if activity is not None and activity < 20.0:
            candidates.append({
                'priority': 2,
                'weight': 20.0 - activity,
                'suggestion': "Consider adding a short walk or another comfortable form of movement to your day.",
                'reason': f"Reported daily physical movement ({int(activity)} min) is relatively low.",
                'source_signal': "Physical Activity"
            })

    # -------------------------------------------------------------
    # 2. MOOD SIGNALS (Only if recent mood entries exist)
    # -------------------------------------------------------------
    if recent_mood_summary:
        avg_mood = recent_mood_summary.get('avg_rating') if recent_mood_summary.get('avg_rating') is not None else recent_mood_summary.get('avg_7d')
        has_entries = (recent_mood_summary.get('count', 0) > 0) or (recent_mood_summary.get('avg_7d') is not None)
        if has_entries and avg_mood is not None and avg_mood < 3.0:
            candidates.append({
                'priority': 2,
                'weight': 3.0 - avg_mood,
                'suggestion': "Consider planning one small activity that helps you feel connected or supported.",
                'reason': f"Your recent mood check-ins average ({avg_mood:.1f}/5) is on the lower side.",
                'source_signal': "Mood Check-ins"
            })

    # -------------------------------------------------------------
    # 3. JOURNAL REFLECTION SIGNALS (Only if recent journals exist)
    # -------------------------------------------------------------
    if recent_journal_signals and recent_journal_signals.get('count', 0) > 0:
        sentiment = recent_journal_signals.get('avg_sentiment')
        neg_ratio = recent_journal_signals.get('neg_emotion_ratio')
        
        has_neg_sentiment = sentiment is not None and sentiment <= -0.10
        has_high_neg_emotion = neg_ratio is not None and neg_ratio >= 0.40

        if has_neg_sentiment or has_high_neg_emotion:
            candidates.append({
                'priority': 2,
                'weight': (neg_ratio or 0.0) + (abs(sentiment) if (sentiment and sentiment < 0) else 0.0),
                'suggestion': "Your recent reflections contain some difficult themes. You might try writing down one thing within your control today.",
                'reason': "Recent reflections reflect noticeable emotional strain.",
                'source_signal': "Journal Reflections"
            })

    # -------------------------------------------------------------
    # 4. GOAL & HABIT CONSISTENCY (If progress exists)
    # -------------------------------------------------------------
    has_active_progress = False
    if active_goals:
        for g in active_goals:
            if hasattr(g, 'progress_percentage') and g.progress_percentage > 0:
                has_active_progress = True
                break
            elif isinstance(g, dict) and g.get('progress_percentage', 0) > 0:
                has_active_progress = True
                break

    if today_habits_completed > 0 or completed_goals_count > 0 or has_active_progress:
        candidates.append({
            'priority': 3,
            'weight': float(today_habits_completed + completed_goals_count),
            'suggestion': "You're building consistency. Keep the next step small and realistic.",
            'reason': "You have active goals or recent habit completions underway.",
            'source_signal': "Habit & Goal Consistency"
        })

    # -------------------------------------------------------------
    # 5. PRIORITIZE & SELECT TOP 3
    # -------------------------------------------------------------
    candidates.sort(key=lambda x: (x['priority'], -x.get('weight', 0)))

    seen_signals = set()
    filtered = []
    for item in candidates:
        if item['source_signal'] not in seen_signals:
            seen_signals.add(item['source_signal'])
            filtered.append({
                'suggestion': item['suggestion'],
                'reason': item['reason'],
                'source_signal': item['source_signal']
            })
        if len(filtered) >= 3:
            break

    if not filtered:
        filtered.append({
            'suggestion': "Keep checking in with yourself and choose one small, realistic action for today.",
            'reason': "Regular self-reflection helps you notice patterns early.",
            'source_signal': "Daily Self-Reflection"
        })

    return filtered
