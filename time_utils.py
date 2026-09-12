import os
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, time, timedelta

APP_TIMEZONE_NAME = os.environ.get('APP_TIMEZONE', 'Asia/Kolkata')

try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = ZoneInfo('Asia/Kolkata')

def get_utc_now():
    return datetime.now(timezone.utc)

def get_local_now():
    return get_utc_now().astimezone(APP_TIMEZONE)

def get_local_today():
    return get_local_now().date()

def utc_to_local(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(APP_TIMEZONE)
    return dt

def get_local_day_utc_range(local_date):
    start_local = datetime.combine(local_date, time.min, tzinfo=APP_TIMEZONE)
    end_local = datetime.combine(local_date, time.max, tzinfo=APP_TIMEZONE)
    # Strip timezone for naive SQLite comparison or keep UTC
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

def format_local_datetime(dt, fmt='%b %d, %Y %I:%M %p'):
    if not dt:
        return ''
    if isinstance(dt, datetime):
        local_dt = utc_to_local(dt)
        return local_dt.strftime(fmt)
    elif hasattr(dt, 'strftime'):
        return dt.strftime(fmt)
    return str(dt)

def format_local_date(dt, fmt='%b %d, %Y'):
    if not dt:
        return ''
    if isinstance(dt, datetime):
        local_dt = utc_to_local(dt)
        return local_dt.strftime(fmt)
    elif hasattr(dt, 'strftime'):
        return dt.strftime(fmt)
    return str(dt)
