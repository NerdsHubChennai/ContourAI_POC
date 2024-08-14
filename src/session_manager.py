import time

class SessionManager:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.last_activity_time = time.time()

    def reset_timer(self):
        self.last_activity_time = time.time()

    def is_session_expired(self):
        return time.time() - self.last_activity_time > self.timeout_seconds