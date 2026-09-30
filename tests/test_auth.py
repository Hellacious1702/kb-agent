"""Tests for auth utilities."""

from app.utils.auth import WorkspaceAuth


class TestWorkspaceAuth:
    """Tests for WorkspaceAuth."""

    def test_create_api_key(self):
        """Test API key creation."""
        auth = WorkspaceAuth()
        full_key, key_prefix, key_hash = auth.create_api_key("Test Key")

        assert full_key.startswith("sk_")
        assert key_prefix == full_key.split("_")[0] + "_" + full_key.split("_")[1]
        assert len(key_hash) > 0
        assert key_hash != full_key

    def test_verify_api_key(self):
        """Test API key verification."""
        auth = WorkspaceAuth()
        full_key, key_prefix, key_hash = auth.create_api_key("Test Key")

        assert auth.verify_api_key(full_key, key_hash)
        assert not auth.verify_api_key("wrong_key", key_hash)
        assert not auth.verify_api_key(full_key, "wrong_hash")

    def test_create_access_token(self):
        """Test JWT token creation."""
        auth = WorkspaceAuth()
        token = auth.create_access_token("workspace-123", "key-456")

        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_token(self):
        """Test JWT token decoding."""
        auth = WorkspaceAuth()
        token = auth.create_access_token("workspace-123", "key-456")
        payload = auth.decode_token(token)

        assert payload is not None
        assert payload["sub"] == "workspace-123"
        assert payload["api_key_id"] == "key-456"

    def test_decode_invalid_token(self):
        """Test decoding invalid token."""
        auth = WorkspaceAuth()
        payload = auth.decode_token("invalid.token.here")
        assert payload is None