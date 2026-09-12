import os
import unittest
import json
import re
from datetime import datetime, timezone, timedelta
import joblib

os.environ['SECRET_KEY'] = 'test_phase3_secret_key'
os.environ['APP_TIMEZONE'] = 'Asia/Kolkata'

from app import app, db, bcrypt, User, Journal, MoodEntry, BurnoutAssessment, get_user_wellness_signals
from ml.predict_wellness_risk import predict_wellness_risk, get_model_pipeline, get_feature_schema

class Phase3WellnessModelTestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app
        self.client = app.test_client()
        
        with self.app.app_context():
            db.create_all()
            
            # Clean prior test users if any
            prior = User.query.filter(User.email.in_(['alice@test.com', 'bob@test.com'])).all()
            for p in prior:
                db.session.delete(p)
            db.session.commit()

            user1 = User(
                username='alice_wellness',
                email='alice@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            user2 = User(
                username='bob_wellness',
                email='bob@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            db.session.add_all([user1, user2])
            db.session.commit()
            self.user1_id = user1.id
            self.user2_id = user2.id

    def tearDown(self):
        with self.app.app_context():
            test_users = User.query.filter(User.email.in_(['alice@test.com', 'bob@test.com'])).all()
            for u in test_users:
                db.session.delete(u)
            db.session.commit()
            db.session.remove()

    def _login(self, client, email='alice@test.com', password='password123'):
        res = client.get('/login')
        token = self._extract_csrf(res.data.decode('utf-8'))
        return client.post('/login', data={
            'email': email,
            'password': password,
            'csrf_token': token
        }, follow_redirects=True)

    def _extract_csrf(self, html):
        pos1 = html.find('name="csrf_token"')
        if pos1 != -1:
            val_start = html.find('value="', pos1) + 7
            val_end = html.find('"', val_start)
            return html[val_start:val_end]
        pos2 = html.find("name='csrf_token'")
        if pos2 != -1:
            val_start = html.find("value='", pos2) + 7
            val_end = html.find("'", val_start)
            return html[val_start:val_end]
        return ''

    def test_01_model_artifacts_exist_and_load(self):
        model_path = os.path.join('models', 'wellness_risk_model.joblib')
        schema_path = os.path.join('models', 'feature_schema.json')
        metrics_path = os.path.join('models', 'model_metrics.json')
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(schema_path))
        self.assertTrue(os.path.exists(metrics_path))
        
        pipeline = get_model_pipeline()
        self.assertIsNotNone(pipeline)
        
        schema = get_feature_schema()
        self.assertEqual(len(schema['feature_names']), 7)
        self.assertIn('sleep_hours', schema['feature_names'])
        self.assertIn('negative_emotion_frequency', schema['feature_names'])
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        self.assertIn('metrics', metrics)
        self.assertGreater(metrics['metrics']['accuracy'], 0.85)
        self.assertEqual(metrics['dataset_type'], 'synthetic_prototype')

    def test_02_predict_wellness_risk_deterministic_score(self):
        low_input = {
            'sleep_hours': 8.0,
            'work_study_hours': 6.5,
            'stress_level': 2.0,
            'physical_activity_minutes': 45.0,
            'mood_rating': 4.5,
            'journal_sentiment': 0.5,
            'negative_emotion_frequency': 0.0
        }
        res_low = predict_wellness_risk(low_input)
        self.assertIn(res_low['risk_label'], ['Low', 'Moderate'])
        self.assertLessEqual(res_low['risk_score'], 33)
        self.assertEqual(res_low['is_diagnostic'], False)
        self.assertIn('disclaimer', res_low)
        self.assertIn('contributing_factors', res_low)

        high_input = {
            'sleep_hours': 4.0,
            'work_study_hours': 12.0,
            'stress_level': 9.0,
            'physical_activity_minutes': 0.0,
            'mood_rating': 1.5,
            'journal_sentiment': -0.7,
            'negative_emotion_frequency': 0.8
        }
        res_high = predict_wellness_risk(high_input)
        self.assertEqual(res_high['risk_label'], 'Elevated')
        self.assertGreaterEqual(res_high['risk_score'], 67)
        self.assertGreater(len(res_high['contributing_factors']), 0)

    def test_03_predict_wellness_risk_missing_signals_imputation(self):
        partial_input = {
            'sleep_hours': 7.0,
            'work_study_hours': 8.0,
            'stress_level': 5.0,
            'physical_activity_minutes': 30.0,
            'mood_rating': None,
            'journal_sentiment': None,
            'negative_emotion_frequency': None
        }
        res = predict_wellness_risk(partial_input)
        self.assertIn(res['risk_label'], ['Low', 'Moderate', 'Elevated'])
        self.assertTrue(0 <= res['risk_score'] <= 100)
        self.assertIn('mood_rating', res['missing_features'])
        self.assertIn('journal_sentiment', res['missing_features'])
        self.assertIn('negative_emotion_frequency', res['missing_features'])

    def test_04_burnout_assessment_model_and_properties(self):
        with self.app.app_context():
            assessment = BurnoutAssessment(
                user_id=self.user1_id,
                sleep_hours=7.5,
                work_study_hours=8.0,
                stress_level=4.0,
                physical_activity_minutes=40.0,
                mood_rating=3.5,
                journal_sentiment=0.25,
                negative_emotion_frequency=0.1,
                risk_score=28,
                risk_level='Low',
                risk_class=0,
                probabilities_json=json.dumps({'low': 0.85, 'moderate': 0.12, 'elevated': 0.03}),
                contributing_factors_json=json.dumps([{'factor': 'Balanced Sleep', 'severity': 'low'}]),
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            db.session.add(assessment)
            db.session.commit()
            
            saved = BurnoutAssessment.query.filter_by(user_id=self.user1_id).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.risk_score, 28)
            self.assertEqual(saved.risk_level, 'Low')
            self.assertEqual(saved.probabilities['low'], 0.85)
            self.assertEqual(len(saved.contributing_factors), 1)
            self.assertEqual(saved.badge_color, '#10b981')

    def test_05_cascade_delete_burnout_assessment(self):
        with self.app.app_context():
            assessment = BurnoutAssessment(
                user_id=self.user1_id,
                sleep_hours=6.0,
                work_study_hours=8.0,
                stress_level=5.0,
                physical_activity_minutes=30.0,
                risk_score=40,
                risk_level='Moderate',
                risk_class=1
            )
            db.session.add(assessment)
            db.session.commit()
            
            user = db.session.get(User, self.user1_id)
            db.session.delete(user)
            db.session.commit()
            
            remaining = BurnoutAssessment.query.filter_by(user_id=self.user1_id).all()
            self.assertEqual(len(remaining), 0)

    def test_06_get_user_wellness_signals_real_data(self):
        with self.app.app_context():
            signals_empty = get_user_wellness_signals(self.user1_id)
            self.assertIsNone(signals_empty['mood_rating'])
            self.assertIsNone(signals_empty['journal_sentiment'])
            self.assertIsNone(signals_empty['negative_emotion_frequency'])
            self.assertFalse(signals_empty['has_mood_data'])
            self.assertFalse(signals_empty['has_journal_data'])
            
            m1 = MoodEntry(user_id=self.user1_id, mood_rating=4, created_at=datetime.now(timezone.utc).replace(tzinfo=None))
            m2 = MoodEntry(user_id=self.user1_id, mood_rating=2, created_at=datetime.now(timezone.utc).replace(tzinfo=None))
            db.session.add_all([m1, m2])
            
            j1 = Journal(
                user_id=self.user1_id,
                title='Hard day',
                content='Tough day today',
                sentiment='Negative',
                sentiment_score=-0.4,
                emotion='sadness'
            )
            j2 = Journal(
                user_id=self.user1_id,
                title='Better evening',
                content='Relaxed in the evening',
                sentiment='Positive',
                sentiment_score=0.6,
                emotion='joy'
            )
            db.session.add_all([j1, j2])
            db.session.commit()
            
            signals = get_user_wellness_signals(self.user1_id)
            self.assertEqual(signals['mood_rating'], 3.0)
            self.assertEqual(signals['journal_sentiment'], 0.1)
            self.assertEqual(signals['negative_emotion_frequency'], 0.5)
            self.assertTrue(signals['has_mood_data'])
            self.assertTrue(signals['has_journal_data'])

    def test_07_unauthenticated_access_blocked(self):
        get_res = self.client.get('/burnout')
        self.assertEqual(get_res.status_code, 302)
        self.assertIn('/login', get_res.headers['Location'])
        
        login_res = self.client.get('/login')
        csrf_token = self._extract_csrf(login_res.data.decode('utf-8'))
        post_res = self.client.post('/burnout', data={'csrf_token': csrf_token, 'sleep_hours': 7.0})
        self.assertEqual(post_res.status_code, 302)
        self.assertIn('/login', post_res.headers['Location'])

    def test_08_csrf_protection_on_assessment(self):
        self._login(self.client, email='alice@test.com')
        res = self.client.post('/burnout', data={
            'sleep_hours': 7.0,
            'work_study_hours': 8.0,
            'stress_level': 5.0,
            'physical_activity_minutes': 30.0
        })
        self.assertEqual(res.status_code, 400)

    def test_09_authenticated_get_and_post_assessment(self):
        self._login(self.client, email='alice@test.com')
        
        get_res = self.client.get('/burnout')
        self.assertEqual(get_res.status_code, 200)
        html = get_res.data.decode('utf-8')
        self.assertIn('Wellness Risk Assessment', html)
        self.assertIn('Important Notice:', html)
        token = self._extract_csrf(html)
        self.assertTrue(token)
        
        post_res = self.client.post('/burnout', data={
            'csrf_token': token,
            'sleep_hours': '7.5',
            'work_study_hours': '7.0',
            'stress_level': '3',
            'physical_activity_minutes': '45'
        }, follow_redirects=True)
        
        self.assertEqual(post_res.status_code, 200)
        post_html = post_res.data.decode('utf-8')
        self.assertIn('Your wellness risk assessment has been calculated', post_html)
        self.assertIn('Lifestyle & Wellness Strain Analysis', post_html)
        
        with self.app.app_context():
            saved = BurnoutAssessment.query.filter_by(user_id=self.user1_id).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.sleep_hours, 7.5)
            self.assertTrue(0 <= saved.risk_score <= 100)

    def test_10_validation_errors_handled(self):
        self._login(self.client, email='alice@test.com')
        get_res = self.client.get('/burnout')
        token = self._extract_csrf(get_res.data.decode('utf-8'))
        
        post_res = self.client.post('/burnout', data={
            'csrf_token': token,
            'sleep_hours': '-2.0',
            'work_study_hours': '8.0',
            'stress_level': '5',
            'physical_activity_minutes': '30'
        }, follow_redirects=True)
        
        self.assertEqual(post_res.status_code, 200)
        html = post_res.data.decode('utf-8')
        self.assertIn('Sleep hours must be between 0 and 24', html)

    def test_11_user_ownership_isolation(self):
        with self.app.app_context():
            assessment = BurnoutAssessment(
                user_id=self.user1_id,
                sleep_hours=7.0,
                work_study_hours=8.0,
                stress_level=5.0,
                physical_activity_minutes=30.0,
                risk_score=45,
                risk_level='Moderate',
                risk_class=1
            )
            db.session.add(assessment)
            db.session.commit()
            assessment_id = assessment.id
            
        client2 = self.app.test_client()
        self._login(client2, email='bob@test.com')
        
        get_res = client2.get('/burnout')
        token = self._extract_csrf(get_res.data.decode('utf-8'))
        
        del_res = client2.post(f'/burnout/{assessment_id}/delete', data={
            'csrf_token': token
        }, follow_redirects=True)
        
        del_html = del_res.data.decode('utf-8')
        self.assertIn('Unauthorized action', del_html)
        
        with self.app.app_context():
            still_there = db.session.get(BurnoutAssessment, assessment_id)
            self.assertIsNotNone(still_there)

    def test_12_dashboard_and_profile_display_assessment(self):
        with self.app.app_context():
            assessment = BurnoutAssessment(
                user_id=self.user1_id,
                sleep_hours=8.0,
                work_study_hours=6.0,
                stress_level=2.0,
                physical_activity_minutes=50.0,
                risk_score=15,
                risk_level='Low',
                risk_class=0,
                probabilities_json=json.dumps({'low': 0.95, 'moderate': 0.05, 'elevated': 0.0}),
                contributing_factors_json=json.dumps([])
            )
            db.session.add(assessment)
            db.session.commit()

        self._login(self.client, email='alice@test.com')
        
        j_res = self.client.get('/journals')
        self.assertEqual(j_res.status_code, 200)
        j_html = j_res.data.decode('utf-8')
        self.assertIn('AI Wellness Risk Assessment', j_html)
        self.assertIn('15/100', j_html)
        self.assertIn('/burnout', j_html)
        
        p_res = self.client.get('/profile')
        self.assertEqual(p_res.status_code, 200)
        p_html = p_res.data.decode('utf-8')
        self.assertIn('Strain Score', p_html)
        self.assertIn('15', p_html)

if __name__ == '__main__':
    unittest.main(verbosity=2)
