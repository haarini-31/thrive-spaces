import os
import unittest
import json
import re

os.environ['SECRET_KEY'] = 'test_phase5_secret_key'
os.environ['APP_TIMEZONE'] = 'Asia/Kolkata'

from app import app, db, bcrypt, User, BurnoutAssessment
from services.support_resources import (
    get_support_resources,
    get_support_resources_by_category,
    EMERGENCY_CONTACTS,
    RESPONSIBLE_AI_SUPPORT_DISCLAIMER
)

class Phase5SupportResourcesTestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = True
        self.app = app
        self.client = app.test_client()

        with self.app.app_context():
            # Clean prior test users if any
            prior = User.query.filter(User.email.in_(['alice_p5@test.com', 'bob_p5@test.com'])).all()
            for p in prior:
                db.session.delete(p)
            db.session.commit()

            user1 = User(
                username='alice_p5',
                email='alice_p5@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            user2 = User(
                username='bob_p5',
                email='bob_p5@test.com',
                password=bcrypt.generate_password_hash('password123').decode('utf-8')
            )
            db.session.add_all([user1, user2])
            db.session.commit()
            self.user1_id = user1.id
            self.user2_id = user2.id

    def tearDown(self):
        with self.app.app_context():
            test_users = User.query.filter(User.email.in_(['alice_p5@test.com', 'bob_p5@test.com'])).all()
            for u in test_users:
                db.session.delete(u)
            db.session.commit()
            db.session.remove()

    def _login(self, email='alice_p5@test.com', password='password123'):
        res = self.client.get('/login')
        match = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode('utf-8'))
        csrf_token = match.group(1) if match else ''
        return self.client.post('/login', data={
            'email': email,
            'password': password,
            'csrf_token': csrf_token
        }, follow_redirects=True)

    # -------------------------------------------------------------
    # 1. ANONYMOUS ACCESS
    # -------------------------------------------------------------
    def test_01_support_page_accessible_anonymously(self):
        """Verify /support is accessible without login."""
        res = self.client.get('/support')
        self.assertEqual(res.status_code, 200)
        content = res.data.decode('utf-8')
        self.assertIn('Professional Support', content)
        self.assertIn('Need Immediate Help?', content)
        self.assertIn('112', content)

    def test_02_therapists_alias_accessible_anonymously(self):
        """Verify /therapists route also serves the support directory cleanly."""
        res = self.client.get('/therapists')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Professional Support', res.data)

    def test_03_no_private_user_data_exposed_anonymously(self):
        """Verify anonymous access reveals no user details or personalized signals."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8')
        self.assertNotIn('alice_p5', content)
        self.assertNotIn('bob_p5', content)
        self.assertNotIn('My Journals', content)
        self.assertIn('Sign In', content)
        self.assertIn('Create Account', content)

    # -------------------------------------------------------------
    # 2. LOCAL RESOURCE DIRECTORY INTEGRITY
    # -------------------------------------------------------------
    def test_04_support_resources_service_integrity(self):
        """Verify service returns non-empty curated resources with all expected keys."""
        resources = get_support_resources()
        self.assertGreaterEqual(len(resources), 5)
        self.assertLessEqual(len(resources), 15)

        for r in resources:
            self.assertIn('id', r)
            self.assertIn('name', r)
            self.assertIn('category', r)
            self.assertIn('description', r)
            self.assertIn('website', r)
            self.assertIn('region', r)
            self.assertTrue(r['name'].strip())
            self.assertTrue(r['description'].strip())
            self.assertTrue(r['website'].startswith('http'))

    def test_05_support_resources_categories_rendered(self):
        """Verify all major resource categories are present in the service output."""
        cats = get_support_resources_by_category()
        self.assertIn('Crisis / Immediate Help', cats)
        self.assertIn('Mental Wellness Helplines', cats)
        self.assertIn('Professional Support', cats)

    def test_06_support_page_renders_resources(self):
        """Verify the template renders the curated resources."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8')
        self.assertIn('Tele-MANAS', content)
        self.assertIn('KIRAN Helpline', content)
        self.assertIn('Vandrevala Foundation', content)
        self.assertIn('NIMHANS', content)
        self.assertIn('AASRA', content)

    # -------------------------------------------------------------
    # 3. EMERGENCY SAFETY INFORMATION
    # -------------------------------------------------------------
    def test_07_emergency_safety_card_present(self):
        """Verify emergency numbers (112, 14416, 1800-599-0019) are prominent."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8')
        self.assertIn('112', content)
        self.assertIn('14416', content)
        self.assertIn('1800-599-0019', content)
        self.assertIn('Need Immediate Help?', content)
        self.assertIn('tel:112', content)
        self.assertIn('tel:14416', content)

    # -------------------------------------------------------------
    # 4. RESPONSIBLE-AI & NON-CLINICAL SAFEGUARDS
    # -------------------------------------------------------------
    def test_08_responsible_ai_disclaimer_rendered(self):
        """Verify non-clinical disclaimer is displayed."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8')
        self.assertIn('not a substitute', content)
        self.assertIn('qualified mental-health professional', content)
        self.assertIn('prototypes intended for self-reflection', content)

    def test_09_no_diagnostic_or_coercive_claims_in_ui(self):
        """Verify absence of diagnostic or coercive therapy language."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8').lower()
        forbidden_phrases = [
            'you have depression',
            'you have anxiety',
            'clinically at risk',
            'you need treatment',
            'requires therapy'
        ]
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, content)

    # -------------------------------------------------------------
    # 5. PERSONALIZED SUPPORT CTA (AUTHENTICATED)
    # -------------------------------------------------------------
    def test_10_logged_in_user_without_assessment_cta(self):
        """Verify logged-in user without assessment receives gentle wellbeing CTA."""
        self._login('alice_p5@test.com', 'password123')
        res = self.client.get('/support')
        self.assertEqual(res.status_code, 200)
        content = res.data.decode('utf-8')
        self.assertIn('Professional support is available if you would like help understanding concerns', content)
        self.assertIn('My Journals', content)
        self.assertIn('Logout', content)

    def test_11_logged_in_user_with_elevated_assessment_cta(self):
        """Verify user with moderate/elevated risk receives appropriate non-clinical recommendation."""
        self._login('alice_p5@test.com', 'password123')

        # Insert elevated assessment for alice
        with self.app.app_context():
            assessment = BurnoutAssessment(
                user_id=self.user1_id,
                sleep_hours=4.5,
                work_study_hours=11.0,
                stress_level=8.5,
                physical_activity_minutes=15.0,
                risk_score=78,
                risk_level='Elevated',
                risk_class=2,
                probabilities_json=json.dumps({'Low': 0.1, 'Moderate': 0.2, 'Elevated': 0.7}),
                contributing_factors_json=json.dumps([])
            )
            db.session.add(assessment)
            db.session.commit()

        res = self.client.get('/support')
        self.assertEqual(res.status_code, 200)
        content = res.data.decode('utf-8')
        self.assertIn('Consider speaking with a qualified professional if these concerns are persistent', content)
        self.assertNotIn('you must see a doctor', content.lower())

    # -------------------------------------------------------------
    # 6. SECURITY & FREE-ONLY BEHAVIOR
    # -------------------------------------------------------------
    def test_12_external_links_use_safe_attributes(self):
        """Verify all external website links have target=_blank and rel=noopener noreferrer."""
        res = self.client.get('/support')
        content = res.data.decode('utf-8')
        links = re.findall(r'<a\s+[^>]*href="(https?://[^"]+)"[^>]*>', content)
        self.assertGreater(len(links), 0)

        # Check raw tags with regex
        raw_anchor_tags = re.findall(r'(<a\s+[^>]*href="https?://[^"]+"[^>]*>)', content)
        for tag in raw_anchor_tags:
            self.assertIn('target="_blank"', tag)
            self.assertIn('rel="noopener noreferrer"', tag)

    def test_13_no_api_key_dependency(self):
        """Verify support page and get-therapists route work even if GOOGLE_MAPS_API_KEY is empty."""
        old_val = os.environ.get('GOOGLE_MAPS_API_KEY')
        try:
            if 'GOOGLE_MAPS_API_KEY' in os.environ:
                del os.environ['GOOGLE_MAPS_API_KEY']

            res_page = self.client.get('/support')
            self.assertEqual(res_page.status_code, 200)

            res_api = self.client.get('/get-therapists')
            self.assertEqual(res_api.status_code, 200)
            data = json.loads(res_api.data.decode('utf-8'))
            self.assertEqual(data['status'], 'success')
            self.assertGreater(len(data['results']), 0)
        finally:
            if old_val is not None:
                os.environ['GOOGLE_MAPS_API_KEY'] = old_val

    def test_14_authenticated_pages_remain_protected(self):
        """Verify core private routes still require login."""
        protected_routes = ['/journals', '/create-journal', '/mood', '/mood/history', '/burnout', '/goals', '/profile']
        for route in protected_routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 302, f"Route {route} should redirect anonymous users")
            self.assertIn('/login', res.location)

if __name__ == '__main__':
    unittest.main()
