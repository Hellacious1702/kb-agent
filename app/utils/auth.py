"""Authentication and authorization utilities."""

import secrets
from datetime import datetime, timedelta

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import APIKey, Workspace

settings = get_settings()


class WorkspaceAuth:
    """
    Workspace-level authentication using API keys.
    
    WHY: API keys are simpler than OAuth for internal tools. Each workspace
    gets its own API keys with prefixes for identification. Keys are hashed
    with bcrypt for storage. JWT tokens are used for session-based web UI auth.
    """

    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.token_expire_minutes = settings.access_token_expire_minutes

    def create_api_key(self, name: str) -> tuple[str, str, str]:
        """
        Create a new API key.
        
        Returns:
            Tuple of (full_key, key_prefix, key_hash)
        """
        # Generate random key: sk_{prefix}_{random}
        prefix = secrets.token_urlsafe(8)
        random_part = secrets.token_urlsafe(32)
        full_key = f"sk_{prefix}_{random_part}"
        key_prefix = f"sk_{prefix}"

        # Hash with bcrypt
        key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()

        return full_key, key_prefix, key_hash

    def verify_api_key(self, provided_key: str, key_hash: str) -> bool:
        """Verify an API key against its hash."""
        try:
            return bcrypt.checkpw(provided_key.encode(), key_hash.encode())
        except Exception:
            return False

    def create_access_token(self, workspace_id: str, api_key_id: str) -> str:
        """Create JWT access token for workspace."""
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        payload = {
            "sub": workspace_id,
            "api_key_id": api_key_id,
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict | None:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    async def authenticate_api_key(
        self,
        session: AsyncSession,
        api_key: str,
    ) -> Workspace | None:
        """
        Authenticate an API key and return the workspace.
        
        WHY: We need to find the workspace by checking all active API keys.
        This is O(n) but acceptable for internal tools with limited keys.
        For scale, we'd add a lookup table or use prefix-based partitioning.
        """
        # Extract prefix to narrow search
        if not api_key.startswith("sk_"):
            return None

        prefix_end = api_key.find("_", 3)
        if prefix_end == -1:
            return None
        key_prefix = api_key[:prefix_end + 1]  # Include the underscore

        # Find matching API key
        result = await session.execute(
            select(APIKey).where(
                APIKey.key_prefix == key_prefix,
                APIKey.is_active == True,
            )
        )
        api_keys = result.scalars().all()

        for key in api_keys:
            if self.verify_api_key(api_key, key.key_hash):
                # Check expiration
                if key.expires_at and key.expires_at < datetime.utcnow():
                    return None

                # Update last used
                key.last_used_at = datetime.utcnow()
                await session.commit()

                # Get workspace
                workspace_result = await session.execute(
                    select(Workspace).where(Workspace.id == key.workspace_id)
                )
                return workspace_result.scalar_one_or_none()

        return None

    async def authenticate_token(
        self,
        session: AsyncSession,
        token: str,
    ) -> Workspace | None:
        """Authenticate a JWT token and return the workspace."""
        payload = self.decode_token(token)
        if not payload:
            return None

        workspace_id = payload.get("sub")
        if not workspace_id:
            return None

        result = await session.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        return result.scalar_one_or_none()


# Global auth instance
workspace_auth = WorkspaceAuth()
