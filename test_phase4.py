import os
import unittest
import json
import re
from datetime import datetime, timezone, timedelta

os.environ['SECRET_KEY'] = 'test_phase4_secret_key'
os.environ['APP_TIMEZONE'] = 'Asia/Kolkata'

from app import (
    app, db, bcrypt, User, Journal, MoodEntry, BurnoutAssessment,
    Goal, Habit, HabitLog,
    VALID_GOAL_CATEGORIES, VALID_HABIT_CATEGORIES, VALID_HABIT_FREQUENCIES,
    get_habit_stats_for_user
)
from services.wellness_suggestions import generate_wellness_suggestions, SUGGESTION_DISCLAIMER
from time_utils import get_local_today

class Phase4GoalsHabitsSuggestionsTestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = True
        self.app = app
        self.client = app.test_client()

        with self.app.app_context():
            # Clean prior test users if any
            prior = User.query.filter(User.email.in_(['alice_p4@test.com', 'bob_p4@test.com'])).all()
            for p in prior:
                db.session.delete(p)
            db.session.commit()

            user1 = User(
                username='alice_p4',
                email='alice_p4@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            user2 = User(
                username='bob_p4',
                email='bob_p4@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            db.session.add_all([user1, user2])
            db.session.commit()
            self.user1_id = user1.id
            self.user2_id = user2.id

    def tearDown(self):
        with self.app.app_context():
            test_users = User.query.filter(User.email.in_(['alice_p4@test.com', 'bob_p4@test.com'])).all()
            for u in test_users:
                db.session.delete(u)
            db.session.commit()
            db.session.remove()

    def _get_csrf_token(self, client, url='/login'):
        res = client.get(url)
        match = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode('utf-8'))
        if match:
            return match.group(1)
        match = re.search(r'value="([^"]+)"\s+name="csrf_token"', res.data.decode('utf-8'))
        return match.group(1) if match else None

    def _login(self, client, email='alice_p4@test.com', password='password123'):
        token = self._get_csrf_token(client, '/login')
        return client.post('/login', data={
            'email': email,
            'password': password,
            'csrf_token': token
        }, follow_redirects=True)

    # -------------------------------------------------------------
    # 1. ANONYMOUS ACCESS BLOCKED
    # -------------------------------------------------------------
    def test_01_goals_anonymous_access_blocked(self):
        client = self.app.test_client()
        token = self._get_csrf_token(client, '/login')

        res = client.get('/goals')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers.get('Location', ''))

        res = client.post('/goals/create', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/goals/1/edit', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/goals/1/complete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/goals/1/delete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/habits/create', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/habits/1/toggle', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

        res = client.post('/habits/1/delete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 302)

    # -------------------------------------------------------------
    # 2. GOAL CREATION
    # -------------------------------------------------------------
    def test_02_goal_creation_authenticated(self):
        client = self.app.test_client()
        self._login(client)

        token = self._get_csrf_token(client, '/goals')
        res = client.post('/goals/create', data={
            'title': 'Sleep 8 Hours',
            'category': 'Sleep',
            'target_value': '7.0',
            'current_value': '0.0',
            'unit': 'days/week',
            'deadline': '2026-12-31',
            'description': 'Protect restful sleep hours',
            'csrf_token': token
        }, follow_redirects=True)

        self.assertEqual(res.status_code, 200)
        with self.app.app_context():
            goal = Goal.query.filter_by(user_id=self.user1_id, title='Sleep 8 Hours').first()
            self.assertIsNotNone(goal)
            self.assertEqual(goal.category, 'Sleep')
            self.assertEqual(goal.target_value, 7.0)
            self.assertEqual(goal.current_value, 0.0)
            self.assertEqual(goal.status, 'active')
            self.assertFalse(goal.is_completed)
            self.assertEqual(goal.progress_percentage, 0.0)

    # -------------------------------------------------------------
    # 3. GOAL VALIDATION
    # -------------------------------------------------------------
    def test_03_goal_validation(self):
        client = self.app.test_client()
        self._login(client)

        token = self._get_csrf_token(client, '/goals')
        # Empty title
        res = client.post('/goals/create', data={
            'title': '   ',
            'target_value': '5.0',
            'csrf_token': token
        }, follow_redirects=True)
        self.assertIn(b'Goal title must be at least 2 characters', res.data)

        # Non-positive target
        token = self._get_csrf_token(client, '/goals')
        res = client.post('/goals/create', data={
            'title': 'Invalid Target',
            'target_value': '0.0',
            'csrf_token': token
        }, follow_redirects=True)
        self.assertIn(b'Target value must be greater than 0', res.data)

        token = self._get_csrf_token(client, '/goals')
        res = client.post('/goals/create', data={
            'title': 'Negative Target',
            'target_value': '-5.0',
            'csrf_token': token
        }, follow_redirects=True)
        self.assertIn(b'Target value must be greater than 0', res.data)

    # -------------------------------------------------------------
    # 4. GOAL PROGRESS
    # -------------------------------------------------------------
    def test_04_goal_progress_clamping(self):
        with self.app.app_context():
            g1 = Goal(user_id=self.user1_id, title='Walk 10k steps', target_value=10.0, current_value=2.5)
            g2 = Goal(user_id=self.user1_id, title='Exceed target', target_value=5.0, current_value=7.5)
            g3 = Goal(user_id=self.user1_id, title='Completed zero target', target_value=0.0, current_value=0.0, status='completed')

            self.assertEqual(g1.progress_percentage, 25.0)
            self.assertEqual(g2.progress_percentage, 100.0)
            self.assertEqual(g3.progress_percentage, 100.0)

    # -------------------------------------------------------------
    # 5. GOAL COMPLETION
    # -------------------------------------------------------------
    def test_05_goal_completion(self):
        client = self.app.test_client()
        self._login(client)

        with self.app.app_context():
            goal = Goal(user_id=self.user1_id, title='Morning Meditation', target_value=5.0, current_value=2.0)
            db.session.add(goal)
            db.session.commit()
            goal_id = goal.id

        token = self._get_csrf_token(client, '/goals')
        res = client.post(f'/goals/{goal_id}/complete', data={'csrf_token': token}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        with self.app.app_context():
            saved = db.session.get(Goal, goal_id)
            self.assertEqual(saved.status, 'completed')
            self.assertEqual(saved.current_value, 5.0)
            self.assertTrue(saved.is_completed)

        # Re-open
        token = self._get_csrf_token(client, '/goals')
        client.post(f'/goals/{goal_id}/complete', data={'csrf_token': token}, follow_redirects=True)
        with self.app.app_context():
            saved = db.session.get(Goal, goal_id)
            self.assertEqual(saved.status, 'active')

    # -------------------------------------------------------------
    # 6. GOAL DELETION
    # -------------------------------------------------------------
    def test_06_goal_deletion(self):
        client = self.app.test_client()
        self._login(client)

        with self.app.app_context():
            goal = Goal(user_id=self.user1_id, title='Temporary Goal', target_value=3.0)
            db.session.add(goal)
            db.session.commit()
            goal_id = goal.id

        token = self._get_csrf_token(client, '/goals')
        res = client.post(f'/goals/{goal_id}/delete', data={'csrf_token': token}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        with self.app.app_context():
            self.assertIsNone(db.session.get(Goal, goal_id))

    # -------------------------------------------------------------
    # 7. GOAL OWNERSHIP ISOLATION
    # -------------------------------------------------------------
    def test_07_goal_ownership_isolation(self):
        with self.app.app_context():
            goal = Goal(user_id=self.user1_id, title='Alice Secret Goal', target_value=5.0)
            db.session.add(goal)
            db.session.commit()
            goal_id = goal.id

        # Login as Bob
        client_bob = self.app.test_client()
        self._login(client_bob, email='bob_p4@test.com', password='password123')

        token = self._get_csrf_token(client_bob, '/goals')
        # Attempt edit
        res = client_bob.post(f'/goals/{goal_id}/edit', data={'title': 'Hacked', 'csrf_token': token})
        self.assertEqual(res.status_code, 404)

        # Attempt complete
        res = client_bob.post(f'/goals/{goal_id}/complete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 404)

        # Attempt delete
        res = client_bob.post(f'/goals/{goal_id}/delete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 404)

        with self.app.app_context():
            intact = db.session.get(Goal, goal_id)
            self.assertIsNotNone(intact)
            self.assertEqual(intact.title, 'Alice Secret Goal')

    # -------------------------------------------------------------
    # 8. HABIT CREATION
    # -------------------------------------------------------------
    def test_08_habit_creation(self):
        client = self.app.test_client()
        self._login(client)

        token = self._get_csrf_token(client, '/goals')
        res = client.post('/habits/create', data={
            'name': 'Drink 2L Water',
            'category': 'Self-Care',
            'frequency': 'Daily',
            'target_count': '1',
            'csrf_token': token
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        with self.app.app_context():
            habit = Habit.query.filter_by(user_id=self.user1_id, name='Drink 2L Water').first()
            self.assertIsNotNone(habit)
            self.assertEqual(habit.category, 'Self-Care')
            self.assertEqual(habit.frequency, 'Daily')

    # -------------------------------------------------------------
    # 9. HABIT VALIDATION
    # -------------------------------------------------------------
    def test_09_habit_validation(self):
        client = self.app.test_client()
        self._login(client)

        token = self._get_csrf_token(client, '/goals')
        res = client.post('/habits/create', data={
            'name': ' ',
            'category': 'Self-Care',
            'csrf_token': token
        }, follow_redirects=True)
        self.assertIn(b'Habit name must be at least 2 characters', res.data)

    # -------------------------------------------------------------
    # 10. DAILY HABIT COMPLETION
    # -------------------------------------------------------------
    def test_10_daily_habit_completion(self):
        client = self.app.test_client()
        self._login(client)

        with self.app.app_context():
            habit = Habit(user_id=self.user1_id, name='Evening Reading', category='Self-Care', frequency='Daily')
            db.session.add(habit)
            db.session.commit()
            habit_id = habit.id

        token = self._get_csrf_token(client, '/goals')
        res = client.post(f'/habits/{habit_id}/toggle', data={'csrf_token': token}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        today = get_local_today()
        with self.app.app_context():
            log = HabitLog.query.filter_by(habit_id=habit_id, completed_date=today).first()
            self.assertIsNotNone(log)
            self.assertTrue(log.completed)

    # -------------------------------------------------------------
    # 11. DUPLICATE PREVENTION & TOGGLE
    # -------------------------------------------------------------
    def test_11_habit_duplicate_prevention_and_toggle(self):
        client = self.app.test_client()
        self._login(client)

        with self.app.app_context():
            habit = Habit(user_id=self.user1_id, name='Daily Walk', category='Exercise', frequency='Daily')
            db.session.add(habit)
            db.session.commit()
            habit_id = habit.id

        token = self._get_csrf_token(client, '/goals')
        # First toggle: Complete
        client.post(f'/habits/{habit_id}/toggle', data={'csrf_token': token}, follow_redirects=True)

        today = get_local_today()
        with self.app.app_context():
            self.assertEqual(HabitLog.query.filter_by(habit_id=habit_id, completed_date=today).count(), 1)

        # Second toggle: Un-complete
        token = self._get_csrf_token(client, '/goals')
        client.post(f'/habits/{habit_id}/toggle', data={'csrf_token': token}, follow_redirects=True)

        with self.app.app_context():
            self.assertEqual(HabitLog.query.filter_by(habit_id=habit_id, completed_date=today).count(), 0)

    # -------------------------------------------------------------
    # 12. HABIT OWNERSHIP ISOLATION
    # -------------------------------------------------------------
    def test_12_habit_ownership_isolation(self):
        with self.app.app_context():
            habit = Habit(user_id=self.user1_id, name='Alice Meditation')
            db.session.add(habit)
            db.session.commit()
            habit_id = habit.id

        # Bob attempts toggle and delete
        client_bob = self.app.test_client()
        self._login(client_bob, email='bob_p4@test.com', password='password123')

        token = self._get_csrf_token(client_bob, '/goals')
        res = client_bob.post(f'/habits/{habit_id}/toggle', data={'csrf_token': token})
        self.assertEqual(res.status_code, 404)

        res = client_bob.post(f'/habits/{habit_id}/delete', data={'csrf_token': token})
        self.assertEqual(res.status_code, 404)

        with self.app.app_context():
            self.assertIsNotNone(db.session.get(Habit, habit_id))

    # -------------------------------------------------------------
    # 13. SUGGESTIONS: HIGH STRESS
    # -------------------------------------------------------------
    def test_13_suggestion_high_stress(self):
        class MockAssessment:
            stress_level = 8.5
            sleep_hours = 7.0
            work_study_hours = 6.0
            physical_activity_minutes = 30.0

        suggestions = generate_wellness_suggestions(latest_assessment=MockAssessment())
        self.assertGreaterEqual(len(suggestions), 1)
        stress_sugg = next((s for s in suggestions if s['source_signal'] == 'Perceived Stress'), None)
        self.assertIsNotNone(stress_sugg)
        self.assertIn('short pause', stress_sugg['suggestion'].lower())

    # -------------------------------------------------------------
    # 14. SUGGESTIONS: LOW SLEEP
    # -------------------------------------------------------------
    def test_14_suggestion_low_sleep(self):
        class MockAssessment:
            stress_level = 4.0
            sleep_hours = 4.5
            work_study_hours = 6.0
            physical_activity_minutes = 30.0

        suggestions = generate_wellness_suggestions(latest_assessment=MockAssessment())
        self.assertGreaterEqual(len(suggestions), 1)
        sleep_sugg = next((s for s in suggestions if s['source_signal'] == 'Sleep Duration'), None)
        self.assertIsNotNone(sleep_sugg)
        self.assertIn('wind-down', sleep_sugg['suggestion'].lower())

    # -------------------------------------------------------------
    # 15. SUGGESTIONS: LOW MOOD
    # -------------------------------------------------------------
    def test_15_suggestion_low_mood(self):
        mood_summary = {'count': 5, 'avg_rating': 2.1}
        suggestions = generate_wellness_suggestions(recent_mood_summary=mood_summary)
        mood_sugg = next((s for s in suggestions if s['source_signal'] == 'Mood Check-ins'), None)
        self.assertIsNotNone(mood_sugg)
        self.assertIn('connected or supported', mood_sugg['suggestion'].lower())

    # -------------------------------------------------------------
    # 16. SUGGESTIONS: MISSING SIGNALS (NO FABRICATED DATA)
    # -------------------------------------------------------------
    def test_16_suggestion_missing_signals_fallback(self):
        # All signals missing/empty
        suggestions = generate_wellness_suggestions(
            latest_assessment=None,
            recent_mood_summary={'count': 0, 'avg_rating': None},
            recent_journal_signals={'count': 0, 'avg_sentiment': None}
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]['source_signal'], 'Daily Self-Reflection')
        self.assertIn('checking in with yourself', suggestions[0]['suggestion'].lower())

    # -------------------------------------------------------------
    # 17. SUGGESTION LIMIT & PRIORITIZATION
    # -------------------------------------------------------------
    def test_17_suggestion_max_limit_and_priority(self):
        class HeavyStrainAssessment:
            stress_level = 9.0
            sleep_hours = 4.0
            work_study_hours = 11.0
            physical_activity_minutes = 5.0

        mood_summary = {'count': 7, 'avg_rating': 1.8}
        journal_signals = {'count': 5, 'avg_sentiment': -0.6, 'neg_emotion_ratio': 0.8}

        suggestions = generate_wellness_suggestions(
            latest_assessment=HeavyStrainAssessment(),
            recent_mood_summary=mood_summary,
            recent_journal_signals=journal_signals,
            today_habits_completed=2
        )
        # Strict maximum of 3
        self.assertLessEqual(len(suggestions), 3)
        self.assertGreaterEqual(len(suggestions), 1)

    # -------------------------------------------------------------
    # 18. SUGGESTION EXPLAINABILITY & DISCLAIMER
    # -------------------------------------------------------------
    def test_18_suggestion_explainability_and_disclaimer(self):
        class MockAssessment:
            stress_level = 8.0
            sleep_hours = 5.0
            work_study_hours = 7.0
            physical_activity_minutes = 45.0

        suggestions = generate_wellness_suggestions(latest_assessment=MockAssessment())
        for s in suggestions:
            self.assertIn('suggestion', s)
            self.assertIn('reason', s)
            self.assertIn('source_signal', s)
            self.assertTrue(len(s['suggestion']) > 5)
            self.assertTrue(len(s['reason']) > 5)

        self.assertIn('not medical advice or a diagnosis', SUGGESTION_DISCLAIMER)

    # -------------------------------------------------------------
    # 19. DASHBOARD AND PROFILE INTEGRATION
    # -------------------------------------------------------------
    def test_19_dashboard_and_profile_integration(self):
        client = self.app.test_client()
        self._login(client)

        # Verify /journals dashboard contains Phase 4 elements
        res_journals = client.get('/journals')
        self.assertEqual(res_journals.status_code, 200)
        content_j = res_journals.data.decode('utf-8')
        self.assertIn('Your Next Small Steps', content_j)
        self.assertIn('My Active Goals', content_j)
        self.assertIn('Today\'s Habits', content_j)

        # Verify /profile contains Goals & Habits stats
        res_profile = client.get('/profile')
        self.assertEqual(res_profile.status_code, 200)
        content_p = res_profile.data.decode('utf-8')
        self.assertIn('Goals & Habits Overview', content_p)
        self.assertIn('Active Goals', content_p)
        self.assertIn('Completed Goals', content_p)
        self.assertIn('Active Habits', content_p)
        self.assertIn('Habits Done Today', content_p)

if __name__ == '__main__':
    unittest.main()
