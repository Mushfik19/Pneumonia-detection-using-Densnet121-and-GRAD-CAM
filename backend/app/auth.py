import base64
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, EmailStr, Field

DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'pneumovision.sqlite3'
TOKEN_TTL_HOURS = 24 * 7

def init_auth_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id));''')

def _hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return base64.b64encode(salt + digest).decode()

def _verify_password(password, encoded):
    raw = base64.b64decode(encoded.encode()); salt, digest = raw[:16], raw[16:]
    return hmac.compare_digest(hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1), digest)

def _token_hash(token): return hashlib.sha256(token.encode()).hexdigest()
def _user(row): return {'id': row[0], 'name': row[1], 'email': row[2]}

class RegisterPayload(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
class LoginPayload(BaseModel):
    email: EmailStr
    password: str

def register(payload):
    user_id = secrets.token_urlsafe(16); now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?)', (user_id, payload.name.strip(), payload.email.lower(), _hash_password(payload.password), now))
    except sqlite3.IntegrityError as exc: raise HTTPException(409, 'An account with this email already exists.') from exc
    return {'id': user_id, 'name': payload.name.strip(), 'email': payload.email.lower()}

def login(payload):
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute('SELECT id,name,email,password_hash FROM users WHERE email=?', (payload.email.lower(),)).fetchone()
    if not row or not _verify_password(payload.password, row[3]): raise HTTPException(401, 'Incorrect email or password.')
    token = secrets.token_urlsafe(32); expiry = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)).isoformat()
    with sqlite3.connect(DB_PATH) as db: db.execute('INSERT INTO sessions VALUES (?, ?, ?)', (_token_hash(token), row[0], expiry))
    return token, _user(row)

def current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith('Bearer '): raise HTTPException(401, 'Authentication required.')
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute('SELECT u.id,u.name,u.email FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?', (_token_hash(authorization[7:]), datetime.now(timezone.utc).isoformat())).fetchone()
    if not row: raise HTTPException(401, 'Session is invalid or expired.')
    return _user(row)

def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith('Bearer '):
        with sqlite3.connect(DB_PATH) as db: db.execute('DELETE FROM sessions WHERE token_hash=?', (_token_hash(authorization[7:]),))

