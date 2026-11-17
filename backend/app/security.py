"""Password hashing and signed short-lived access tokens (stdlib only)."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from app import config as _config
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ACCESS_TOKEN_SECONDS = 15 * 60
REFRESH_TOKEN_SECONDS = 30 * 24 * 60 * 60
_configured_secret = os.getenv("FASAL_JWT_SECRET")
_production = (os.getenv("APP_ENV", "development").lower() == "production" or bool(os.getenv("VERCEL")))
if _production and (not _configured_secret or len(_configured_secret.encode("utf-8")) < 32):
    raise RuntimeError("FASAL_JWT_SECRET must be configured in production")
JWT_SECRET = (_configured_secret.encode("utf-8") if _configured_secret else secrets.token_bytes(48))
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${_b64(salt)}${_b64(key)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, expected = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode("utf-8"), salt=_unb64(salt), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(actual, _unb64(expected))
    except (ValueError, TypeError):
        return False


def issue_access_token(user_id: str) -> str:
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({"sub": user_id, "iat": now, "exp": now + ACCESS_TOKEN_SECONDS, "jti": secrets.token_urlsafe(12)}, separators=(",", ":")).encode())
    content = f"{header}.{payload}"
    signature = _b64(hmac.new(JWT_SECRET, content.encode(), hashlib.sha256).digest())
    return f"{content}.{signature}"


def decode_access_token(token: str) -> str:
    try:
        header, payload, signature = token.split(".", 2)
        expected = _b64(hmac.new(JWT_SECRET, f"{header}.{payload}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid token signature")
        claims = json.loads(_unb64(payload))
        if claims.get("exp", 0) <= int(time.time()) or not claims.get("sub"):
            raise ValueError("Expired token")
        return str(claims["sub"])
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token", headers={"WWW-Authenticate": "Bearer"})


def refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    from app.database import get_user_by_id
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})
    token = credentials.credentials
    user = get_user_by_id(decode_access_token(token))
    if user is None:
        raise HTTPException(status_code=401, detail="User account no longer exists", headers={"WWW-Authenticate": "Bearer"})
    return user


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
