"""
ThriveSpace — Professional Support & Wellness Resources
Phase 5: Local, verified, India-first resources, emergency helplines, and safety boundaries.

CRITICAL RULES:
- 100% free and local: zero external paid APIs, zero API key requirements.
- Authoritative, verified resources only (no synthetic or invented organizations).
- Responsible-AI boundaries: non-clinical framing, no diagnostic assertions.
- Safe external linking: all resources include verified official URLs.
"""

RESPONSIBLE_AI_SUPPORT_DISCLAIMER = (
    "ThriveSpace is a wellness and self-reflection tool. Its AI-assisted features are not a substitute "
    "for a qualified mental-health professional. Wellness scores and suggestions are prototypes "
    "intended for self-reflection and should not be interpreted as medical diagnoses."
)

EMERGENCY_CONTACTS = {
    'national_emergency': {
        'name': 'National Emergency Number',
        'number': '112',
        'description': 'All-in-one emergency service across India for immediate medical, police, or rescue assistance.',
        'availability': '24/7 Toll-Free'
    },
    'tele_manas': {
        'name': 'Tele-MANAS (Govt. of India)',
        'number': '14416 / 1800-891-4416',
        'description': 'National Tele Mental Health Programme of India providing 24/7 free psychological support in multiple languages.',
        'availability': '24/7 Toll-Free'
    },
    'kiran': {
        'name': 'KIRAN Mental Health Helpline',
        'number': '1800-599-0019',
        'description': 'National toll-free 24/7 helpline operated by the Ministry of Social Justice & Empowerment for crisis support and psychological first-aid.',
        'availability': '24/7 Toll-Free'
    }
}

SUPPORT_RESOURCES = [
    {
        'id': 'tele-manas',
        'name': 'Tele-MANAS',
        'category': 'Crisis / Immediate Help',
        'description': 'Comprehensive 24/7 national tele-mental health helpline established by the Ministry of Health and Family Welfare (MOHFW), offering free, confidential counseling across regional Indian languages.',
        'phone': '14416 or 1800-891-4416',
        'website': 'https://telemanas.mohfw.gov.in',
        'availability': '24/7, Toll-Free',
        'region': 'India (National)',
        'tags': ['National Helpline', 'Govt of India', 'Multilingual', 'Crisis Support']
    },
    {
        'id': 'kiran-helpline',
        'name': 'KIRAN Helpline',
        'category': 'Mental Wellness Helplines',
        'description': 'National toll-free mental health helpline by the Ministry of Social Justice and Empowerment offering early screening, first aid, emotional support, and mental health referrals.',
        'phone': '1800-599-0019',
        'website': 'https://depwd.gov.in',
        'availability': '24/7, Toll-Free',
        'region': 'India (National)',
        'tags': ['Toll-Free', 'Rehabilitation', 'Screening', 'Counseling']
    },
    {
        'id': 'vandrevala-foundation',
        'name': 'Vandrevala Foundation',
        'category': 'Mental Wellness Helplines',
        'description': 'Round-the-clock mental health support service providing free, confidential counseling and crisis intervention by trained clinical psychologists and counselors.',
        'phone': '+91 9999 666 555',
        'website': 'https://www.vandrevalafoundation.com',
        'availability': '24/7 Helpline',
        'region': 'India (National)',
        'tags': ['Free Counseling', 'Crisis Intervention', '24/7']
    },
    {
        'id': 'aasra',
        'name': 'AASRA',
        'category': 'Crisis / Immediate Help',
        'description': 'A volunteer-driven, non-judgmental emotional support helpline assisting individuals experiencing severe emotional distress, despair, or crisis.',
        'phone': '+91 98204 66726',
        'website': 'http://www.aasra.info',
        'availability': '24/7 Helpline',
        'region': 'India (National)',
        'tags': ['Confidential', 'Emotional Support', 'Crisis Care']
    },
    {
        'id': 'nimhans-wellbeing',
        'name': 'NIMHANS Center for Well-Being',
        'category': 'Professional Support',
        'description': 'The dedicated community extension and wellness arm of the National Institute of Mental Health and Neurosciences (NIMHANS), providing professional counseling, psychotherapy, and positive mental health services.',
        'phone': '080-26995000',
        'website': 'https://nimhans.ac.in',
        'availability': 'Monday – Saturday (Outpatient)',
        'region': 'India (Bengaluru / National Referrals)',
        'tags': ['Premier Institute', 'Clinical Psychologists', 'Consultations']
    },
    {
        'id': 'icall-tiss',
        'name': 'iCALL Psychosocial Helpline (TISS)',
        'category': 'Mental Wellness Helplines',
        'description': 'A pioneering field action project of the Tata Institute of Social Sciences (TISS), offering professional and free psychological counseling via telephone and email.',
        'phone': '9152987821',
        'website': 'https://icallhelpline.org',
        'availability': 'Monday – Saturday: 10:00 AM – 8:00 PM',
        'region': 'India (National)',
        'tags': ['TISS Mumbai', 'Psychosocial Support', 'Certified Counselors']
    },
    {
        'id': 'yourdost',
        'name': 'YourDOST',
        'category': 'Student Support',
        'description': 'An emotional wellness platform providing counseling and guidance tailored for college students, young professionals, and academic communities.',
        'phone': None,
        'website': 'https://yourdost.com',
        'availability': 'Online Platform',
        'region': 'India',
        'tags': ['Student Wellbeing', 'Academic Stress', 'Online Chat']
    },
    {
        'id': 'live-love-laugh',
        'name': 'The Live Love Laugh Foundation',
        'category': 'Self-Help Resources',
        'description': 'A registered charitable trust offering comprehensive psychoeducational resources, stress management guides, mental health awareness materials, and a curated database of verified mental health professionals across India.',
        'phone': None,
        'website': 'https://www.thelivelovelaughfoundation.org',
        'availability': 'Online Educational Resources',
        'region': 'India',
        'tags': ['Self-Help', 'Awareness', 'Directory', 'Education']
    }
]

def get_support_resources():
    """Return all curated support resources."""
    return SUPPORT_RESOURCES

def get_support_resources_by_category():
    """Return support resources grouped by their category."""
    categories = {}
    for resource in SUPPORT_RESOURCES:
        cat = resource['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(resource)
    return categories
