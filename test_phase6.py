import os
import unittest
import json
import re

os.environ['SECRET_KEY'] = 'test_phase6_secure_key_12345'
os.environ['APP_TIMEZONE'] = 'Asia/Kolkata'

from app import app, db, bcrypt, User, Journal, MoodEntry, BurnoutAssessment, Goal, Habit, HabitLog

class Phase6SecurityAndPolishTestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = True
        self.app = app
        self.client = app.test_client()

        with self.app.app_context():
            # Clean prior test users if any
            prior = User.query.filter(User.email.in_(['alice_p6@test.com', 'bob_p6@test.com'])).all()
            for p in prior:
                db.session.delete(p)
            db.session.commit()

            user1 = User(
                username='alice_p6',
                email='alice_p6@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            user2 = User(
                username='bob_p6',
                email='bob_p6@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            db.session.add_all([user1, user2])
            db.session.commit()
            self.user1_id = user1.id
            self.user2_id = user2.id

    def tearDown(self):
        with self.app.app_context():
            test_users = User.query.filter(User.email.in_(['alice_p6@test.com', 'bob_p6@test.com'])).all()
            for u in test_users:
                db.session.delete(u)
            db.session.commit()
            db.session.remove()

    def _get_csrf_token(self, client, url='/login'):
        res = client.get(url)
        match = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode('utf-8'))
        return match.group(1) if match else ''

    def _login(self, client, email='alice_p6@test.com', password='password123'):
        csrf_token = self._get_csrf_token(client, '/login')
        return client.post('/login', data={
            'email': email,
            'password': password,
            'csrf_token': csrf_token
        }, follow_redirects=True)

    # -------------------------------------------------------------
    # 1. ERROR HANDLING TESTS
    # -------------------------------------------------------------
    def test_01_custom_404_error_page(self):
        """Verify 404 responses render the clean user-facing error template."""
        res = self.client.get('/this-page-definitely-does-not-exist-404')
        self.assertEqual(res.status_code, 404)
        content = res.data.decode('utf-8')
        self.assertIn('Page Not Found', content)
        self.assertIn('404', content)
        self.assertIn('Back to Dashboard', content)
        self.assertNotIn('Traceback', content)
        self.assertNotIn('Exception', content)

    def test_02_csrf_failure_error_page(self):
        """Verify submitting a form without CSRF token renders 400 security check template."""
        res = self.client.post('/login', data={
            'email': 'alice_p6@test.com',
            'password': 'password123'
        })
        self.assertEqual(res.status_code, 400)
        content = res.data.decode('utf-8')
        self.assertIn('Session Security Check Failed', content)
        self.assertNotIn('Traceback', content)

    # -------------------------------------------------------------
    # 2. SECRET KEY AUDIT
    # -------------------------------------------------------------
    def test_03_secret_key_not_empty_and_configured(self):
        """Verify SECRET_KEY is actively set and not empty."""
        secret = self.app.config.get('SECRET_KEY')
        self.assertIsNotNone(secret)
        self.assertTrue(len(secret) >= 16)

    # -------------------------------------------------------------
    # 3. STRICT IDOR & AUTHORIZATION CONTROLS
    # -------------------------------------------------------------
    def test_04_idor_protection_burnout_delete(self):
        """Verify User A cannot delete User B's burnout assessment."""
        # Create assessment for User B
        with self.app.app_context():
            assessment_b = BurnoutAssessment(
                user_id=self.user2_id,
                sleep_hours=6.0,
                work_study_hours=8.0,
                stress_level=5.0,
                physical_activity_minutes=30.0,
                risk_score=45,
                risk_level='Moderate',
                risk_class=1,
                probabilities_json=json.dumps({'Low': 0.2, 'Moderate': 0.7, 'Elevated': 0.1}),
                contributing_factors_json=json.dumps([])
            )
            db.session.add(assessment_b)
            db.session.commit()
            b_assessment_id = assessment_b.id

        # Login as User A
        self._login(self.client, 'alice_p6@test.com', 'password123')
        csrf_token = self._get_csrf_token(self.client, '/burnout')

        # Attempt to delete User B's assessment
        res = self.client.post(
            f'/burnout/{b_assessment_id}/delete',
            data={'csrf_token': csrf_token},
            follow_redirects=True
        )
        self.assertIn('Unauthorized action', res.data.decode('utf-8'))

        # Confirm assessment still exists
        with self.app.app_context():
            remaining = BurnoutAssessment.query.filter_by(id=b_assessment_id).first()
            self.assertIsNotNone(remaining)

    def test_05_idor_protection_goal_and_habit_delete(self):
        """Verify User A cannot delete User B's goals or habits."""
        with self.app.app_context():
            goal_b = Goal(user_id=self.user2_id, title='Bob Goal', category='Sleep', target_value=8.0)
            habit_b = Habit(user_id=self.user2_id, name='Bob Habit', category='Exercise')
            db.session.add_all([goal_b, habit_b])
            db.session.commit()
            g_id = goal_b.id
            h_id = habit_b.id

        self._login(self.client, 'alice_p6@test.com', 'password123')
        csrf_token = self._get_csrf_token(self.client, '/goals')

        # Try deleting Bob's goal
        res_g = self.client.post(f'/goals/{g_id}/delete', data={'csrf_token': csrf_token})
        self.assertEqual(res_g.status_code, 404)

        # Try deleting Bob's habit
        res_h = self.client.post(f'/habits/{h_id}/delete', data={'csrf_token': csrf_token})
        self.assertEqual(res_h.status_code, 404)

        # Verify neither was deleted
        with self.app.app_context():
            self.assertIsNotNone(Goal.query.filter_by(id=g_id).first())
            self.assertIsNotNone(Habit.query.filter_by(id=h_id).first())

    # -------------------------------------------------------------
    # 4. CASCADE DELETION INTEGRITY
    # -------------------------------------------------------------
    def test_06_user_cascade_deletion_cleans_all_records(self):
        """Verify deleting a user cascades cleanly and leaves zero orphan rows."""
        with self.app.app_context():
            u = User(username='cascade_user', email='cascade@test.com', password='pw')
            db.session.add(u)
            db.session.commit()
            u_id = u.id

            j = Journal(user_id=u_id, title='T', content='C', sentiment='Neutral', emotion='neutral')
            m = MoodEntry(user_id=u_id, mood_rating=4, mood_tag='Work')
            b = BurnoutAssessment(
                user_id=u_id, sleep_hours=7.0, work_study_hours=7.0, stress_level=4.0,
                physical_activity_minutes=30.0, risk_score=25, risk_level='Low', risk_class=0,
                probabilities_json='{}', contributing_factors_json='[]'
            )
            g = Goal(user_id=u_id, title='Goal', category='Sleep', target_value=8.0)
            h = Habit(user_id=u_id, name='Habit', category='Sleep')
            db.session.add_all([j, m, b, g, h])
            db.session.commit()

            hl = HabitLog(habit_id=h.id, user_id=u_id, completed_date=m.local_date, completed=True)
            db.session.add(hl)
            db.session.commit()

            # Delete the user
            db.session.delete(u)
            db.session.commit()

            # Verify zero orphaned records remain
            self.assertEqual(Journal.query.filter_by(user_id=u_id).count(), 0)
            self.assertEqual(MoodEntry.query.filter_by(user_id=u_id).count(), 0)
            self.assertEqual(BurnoutAssessment.query.filter_by(user_id=u_id).count(), 0)
            self.assertEqual(Goal.query.filter_by(user_id=u_id).count(), 0)
            self.assertEqual(Habit.query.filter_by(user_id=u_id).count(), 0)
            self.assertEqual(HabitLog.query.filter_by(user_id=u_id).count(), 0)

if __name__ == '__main__':
    unittest.main()
