import unittest
from datetime import datetime, timezone, timedelta
from app import app, db, User, Journal, MoodEntry
from time_utils import get_utc_now, get_local_today, get_local_day_utc_range, APP_TIMEZONE

class Phase2MoodTrackingTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()

        with app.app_context():
            db.create_all()
            # Clean prior test users if any
            prior = User.query.filter(User.email.in_(['user_a@example.com', 'user_b@example.com'])).all()
            for p in prior:
                db.session.delete(p)
            db.session.commit()

            from app import bcrypt
            pw_a = bcrypt.generate_password_hash('password_a').decode('utf-8')
            self.user_a = User(username='user_a_mood', email='user_a@example.com', password=pw_a)
            pw_b = bcrypt.generate_password_hash('password_b').decode('utf-8')
            self.user_b = User(username='user_b_mood', email='user_b@example.com', password=pw_b)
            db.session.add_all([self.user_a, self.user_b])
            db.session.commit()

    def tearDown(self):
        with app.app_context():
            test_users = User.query.filter(User.email.in_(['user_a@example.com', 'user_b@example.com'])).all()
            for u in test_users:
                db.session.delete(u)
            db.session.commit()
            db.session.remove()

    def test_01_anonymous_protection(self):
        # Unauthenticated users must be redirected
        res_mood = self.client.get('/mood', follow_redirects=False)
        self.assertEqual(res_mood.status_code, 302)
        self.assertIn('/login', res_mood.location)

        res_history = self.client.get('/mood/history', follow_redirects=False)
        self.assertEqual(res_history.status_code, 302)
        self.assertIn('/login', res_history.location)
        print('PASS 1: Anonymous access protected for /mood and /mood/history')

    def test_02_validation_bounds(self):
        # Log in as User A
        self.client.post('/login', data={'email': 'user_a@example.com', 'password': 'password_a'})

        # Rating below 1 (0)
        res_0 = self.client.post('/mood', data={'mood_rating': '0', 'mood_tag': 'Work / College'}, follow_redirects=True)
        self.assertIn(b'valid mood rating between 1 and 5', res_0.data)

        # Rating above 5 (6)
        res_6 = self.client.post('/mood', data={'mood_rating': '6', 'mood_tag': 'Work / College'}, follow_redirects=True)
        self.assertIn(b'valid mood rating between 1 and 5', res_6.data)

        # Rating invalid text
        res_abc = self.client.post('/mood', data={'mood_rating': 'awesome'}, follow_redirects=True)
        self.assertIn(b'valid mood rating between 1 and 5', res_abc.data)
        print('PASS 2: Server-side validation rejects ratings outside 1-5')

    def test_03_creation_and_duplicate_update(self):
        # Log in as User A
        self.client.post('/login', data={'email': 'user_a@example.com', 'password': 'password_a'})

        # 1st Check-in for today
        res1 = self.client.post('/mood', data={
            'mood_rating': '4',
            'mood_tag': 'Sleep',
            'note': 'Woke up feeling refreshed'
        }, follow_redirects=True)
        self.assertIn(b'daily mood has been recorded', res1.data)

        with app.app_context():
            u = User.query.filter_by(username='user_a_mood').first()
            entries = MoodEntry.query.filter_by(user_id=u.id).all()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].mood_rating, 4)
            self.assertEqual(entries[0].mood_tag, 'Sleep')
            self.assertEqual(entries[0].note, 'Woke up feeling refreshed')
            self.assertIsNotNone(entries[0].created_at)

        # 2nd Check-in for the SAME day -> should UPDATE the record, not create duplicate
        res2 = self.client.post('/mood', data={
            'mood_rating': '5',
            'mood_tag': 'Social',
            'note': 'Had a great lunch with friends'
        }, follow_redirects=True)
        self.assertIn(b'mood check-in has been updated', res2.data)

        with app.app_context():
            u = User.query.filter_by(username='user_a_mood').first()
            entries = MoodEntry.query.filter_by(user_id=u.id).all()
            self.assertEqual(len(entries), 1, 'Duplicate daily entry was incorrectly created!')
            self.assertEqual(entries[0].mood_rating, 5)
            self.assertEqual(entries[0].mood_tag, 'Social')
            self.assertEqual(entries[0].note, 'Had a great lunch with friends')

        print('PASS 3: Creation and single-daily-entry update verified cleanly')

    def test_04_ownership_isolation(self):
        # User A checks in
        self.client.post('/login', data={'email': 'user_a@example.com', 'password': 'password_a'})
        self.client.post('/mood', data={'mood_rating': '5', 'note': 'Private secret note of User A'})
        self.client.get('/logout')

        # User B logs in and views /mood/history
        self.client.post('/login', data={'email': 'user_b@example.com', 'password': 'password_b'})
        res_b_history = self.client.get('/mood/history')
        self.assertNotIn(b'Private secret note of User A', res_b_history.data)
        self.assertIn(b'No check-ins yet', res_b_history.data)

        # User B logs a check-in
        self.client.post('/mood', data={'mood_rating': '2', 'note': 'User B mood note'})

        with app.app_context():
            u_a = User.query.filter_by(username='user_a_mood').first()
            u_b = User.query.filter_by(username='user_b_mood').first()
            self.assertEqual(MoodEntry.query.filter_by(user_id=u_a.id).count(), 1)
            self.assertEqual(MoodEntry.query.filter_by(user_id=u_b.id).count(), 1)

        print('PASS 4: Strict data ownership isolation confirmed between users')

    def test_05_analytics_and_seven_day_summary(self):
        # Populate realistic past entries for User A across several Asia/Kolkata dates
        with app.app_context():
            u = User.query.filter_by(username='user_a_mood').first()
            today_local = get_local_today()

            # Insert ratings for 3 distinct past days: 4, 3, 5
            for days_back, r in [(1, 4), (2, 3), (3, 5)]:
                past_date = today_local - timedelta(days=days_back)
                start_utc, _ = get_local_day_utc_range(past_date)
                m = MoodEntry(
                    user_id=u.id,
                    mood_rating=r,
                    mood_tag='Health',
                    note=f'Past note {days_back}',
                    created_at=start_utc.replace(tzinfo=None)
                )
                db.session.add(m)
            db.session.commit()

        # Login as User A and view dashboard
        self.client.post('/login', data={'email': 'user_a@example.com', 'password': 'password_a'})
        res_dash = self.client.get('/journals')
        self.assertEqual(res_dash.status_code, 200)

        # (4 + 3 + 5) / 3 = 4.0
        self.assertIn(b'4.0 / 5', res_dash.data)
        self.assertIn(b'7-Day Mood Trend', res_dash.data)
        self.assertIn(b'moodChart', res_dash.data)
        print('PASS 5: 7-day average (4.0) and Chart.js integration verified with real database records')

    def test_06_regression_phase1(self):
        self.client.post('/login', data={'email': 'user_a@example.com', 'password': 'password_a'})

        # Journal CRUD
        res_j = self.client.post('/create-journal', data={
            'title': 'Regression Test Journal',
            'content': 'I feel peaceful and hopeful about the future.'
        }, follow_redirects=True)
        self.assertIn(b'Regression Test Journal', res_j.data)
        self.assertIn(b'Positive', res_j.data)

        # Profile
        res_prof = self.client.get('/profile')
        self.assertEqual(res_prof.status_code, 200)
        self.assertIn(b'Activity Summary', res_prof.data)

        # Therapist view
        res_th = self.client.get('/therapists')
        self.assertEqual(res_th.status_code, 200)

        print('PASS 6: Regression verification - Journals, AI NLP, Profile, and Therapists intact')

if __name__ == '__main__':
    unittest.main()
