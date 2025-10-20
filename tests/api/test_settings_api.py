"""
Tests for Settings API endpoints, particularly the secrets CRUD operations.
"""

import pytest
from fastapi.testclient import TestClient
from nodetool.models.secret import Secret
from nodetool.api.settings import SecretCreateRequest, SecretUpdateRequest


@pytest.mark.asyncio
class TestSecretsAPI:
    """Tests for encrypted secrets API endpoints."""

    async def test_list_secrets_empty(self, client: TestClient, headers: dict, user_id: str):
        """
        Test listing secrets when none are configured.
        Verifies that all possible secrets from registry are returned with is_configured=False.
        """
        response = client.get("/api/settings/secrets", headers=headers)
        assert response.status_code == 200
        json_response = response.json()
        # Should return all possible secrets from settings registry
        assert len(json_response["secrets"]) > 0
        # All should be unconfigured initially
        for secret in json_response["secrets"]:
            assert secret["is_configured"] is False
            assert secret["key"] is not None
            assert secret["description"] is not None

    async def test_create_secret(self, client: TestClient, headers: dict, user_id: str):
        """
        Test creating a new secret via PUT (upsert).
        Verifies that the secret is created with encrypted value.
        """
        request_data = {
            "value": "sk-test-12345",
            "description": "OpenAI API key"
        }
        response = client.put("/api/settings/secrets/OPENAI_API_KEY", json=request_data, headers=headers)

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == "OPENAI_API_KEY"
        assert json_response["user_id"] == user_id
        assert json_response["description"] == "OpenAI API key"
        assert json_response["is_configured"] is True
        assert json_response["id"] is not None
        assert json_response["created_at"] is not None
        assert json_response["updated_at"] is not None

    async def test_create_secret_without_description(self, client: TestClient, headers: dict, user_id: str):
        """
        Test creating a secret without description via PUT.
        Verifies that description defaults to registry description.
        """
        request_data = {
            "value": "secret_value"
        }
        response = client.put("/api/settings/secrets/ANTHROPIC_API_KEY", json=request_data, headers=headers)

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == "ANTHROPIC_API_KEY"
        assert json_response["is_configured"] is True
        # Description should be from registry or default
        assert json_response["description"] is not None

    async def test_list_secrets(self, client: TestClient, headers: dict, user_id: str):
        """
        Test listing secrets.
        Verifies that configured secrets are returned in the list with is_configured=True.
        """
        # Configure multiple secrets via PUT
        test_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        for key in test_keys:
            request_data = {
                "value": f"secret_value_for_{key}",
                "description": f"Secret for {key}"
            }
            response = client.put(f"/api/settings/secrets/{key}", json=request_data, headers=headers)
            assert response.status_code == 200

        # List all secrets - will include configured and unconfigured ones
        response = client.get("/api/settings/secrets", headers=headers)
        assert response.status_code == 200
        json_response = response.json()
        # Should have many secrets from registry
        assert len(json_response["secrets"]) > 0

        # Verify the 3 configured secrets are in the list
        configured_secrets = [s for s in json_response["secrets"] if s["key"] in test_keys]
        assert len(configured_secrets) == 3

        for secret in configured_secrets:
            assert secret["key"] in test_keys
            assert secret["user_id"] == user_id
            assert secret["is_configured"] is True
            assert secret["created_at"] is not None
            assert secret["updated_at"] is not None

    async def test_list_secrets_pagination(self, client: TestClient, headers: dict, user_id: str):
        """
        Test pagination when listing secrets.
        Verifies that limit parameter works correctly.
        """
        # First, configure a few secrets to test pagination
        # Using secrets we know exist in the registry
        secret_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "REPLICATE_API_TOKEN"]
        for key in secret_keys:
            request_data = {
                "value": f"value_{key}"
            }
            client.put(f"/api/settings/secrets/{key}", json=request_data, headers=headers)

        # Get all secrets
        response = client.get(
            "/api/settings/secrets",
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        total_secrets = len(json_response["secrets"])
        assert total_secrets > 0

        # Get first page with limit=2
        response = client.get(
            "/api/settings/secrets",
            headers=headers,
            params={"limit": 2}
        )
        assert response.status_code == 200
        json_response = response.json()
        # Should return at most 2 secrets
        assert len(json_response["secrets"]) <= 2
        # If we have more than 2 secrets total, next_key should be set
        if total_secrets > 2:
            assert json_response.get("next_key") is not None or len(json_response["secrets"]) == total_secrets

    async def test_get_secret_not_decrypted(self, client: TestClient, headers: dict, user_id: str):
        """
        Test getting a secret without decryption.
        Verifies that metadata is returned but value is not included.
        """
        # Create a secret via PUT
        secret_key = "OPENAI_API_KEY"
        put_data = {
            "value": "secret_value_12345"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=put_data, headers=headers)
        assert response.status_code == 200

        # Get secret without decryption
        response = client.get(
            f"/api/settings/secrets/{secret_key}",
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == secret_key
        assert json_response["user_id"] == user_id
        assert "value" not in json_response or json_response.get("value") is None

    async def test_get_secret_decrypted(self, client: TestClient, headers: dict, user_id: str):
        """
        Test getting a secret with decryption.
        Verifies that the correct plaintext value is returned.
        """
        # Create a secret via PUT
        secret_key = "ANTHROPIC_API_KEY"
        secret_value = "my_secret_api_key_xyz"
        put_data = {
            "value": secret_value
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=put_data, headers=headers)
        assert response.status_code == 200

        # Get secret with decryption
        response = client.get(
            f"/api/settings/secrets/{secret_key}?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == secret_key
        assert json_response["value"] == secret_value

    async def test_get_secret_not_found(self, client: TestClient, headers: dict):
        """
        Test getting a secret that doesn't exist.
        Verifies that 404 is returned.
        """
        response = client.get(
            "/api/settings/secrets/NONEXISTENT_KEY",
            headers=headers
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_update_secret_value(self, client: TestClient, headers: dict, user_id: str):
        """
        Test updating a secret's value.
        Verifies that the value is updated and metadata is preserved.
        """
        # Create a secret via PUT
        secret_key = "GEMINI_API_KEY"
        create_data = {
            "value": "original_value",
            "description": "Original description"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=create_data, headers=headers)
        assert response.status_code == 200
        original_secret = response.json()

        # Update the secret
        update_data = {
            "value": "updated_value"
        }
        response = client.put(
            f"/api/settings/secrets/{secret_key}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == secret_key
        # Description should be preserved from original
        assert json_response["description"] is not None

        # Verify the new value is encrypted correctly
        response = client.get(
            f"/api/settings/secrets/{secret_key}?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == "updated_value"

    async def test_update_secret_description(self, client: TestClient, headers: dict, user_id: str):
        """
        Test updating a secret's description.
        Verifies that description can be updated.
        """
        # Create a secret via PUT
        secret_key = "HF_TOKEN"
        create_data = {
            "value": "secret_value",
            "description": "Original"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=create_data, headers=headers)
        assert response.status_code == 200

        # Update description only
        update_data = {
            "value": "secret_value",  # Same value
            "description": "Updated description"
        }
        response = client.put(
            f"/api/settings/secrets/{secret_key}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["description"] == "Updated description"

        # Verify value didn't change
        response = client.get(
            f"/api/settings/secrets/{secret_key}?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == "secret_value"

    async def test_update_secret_not_found(self, client: TestClient, headers: dict):
        """
        Test updating a secret that's not in the registry.
        Verifies that 404 is returned for non-registry secrets.
        """
        update_data = {
            "value": "new_value"
        }
        response = client.put(
            "/api/settings/secrets/NONEXISTENT_SECRET_NOT_IN_REGISTRY",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 404
        assert "not available" in response.json()["detail"].lower()

    async def test_delete_secret(self, client: TestClient, headers: dict, user_id: str):
        """
        Test deleting a secret.
        Verifies that the secret is removed from the database.
        """
        # Create a secret via PUT
        secret_key = "REPLICATE_API_TOKEN"
        create_data = {
            "value": "secret_value"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=create_data, headers=headers)
        assert response.status_code == 200

        # Delete the secret
        response = client.delete(
            f"/api/settings/secrets/{secret_key}",
            headers=headers
        )
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"].lower()

        # Verify it's deleted
        response = client.get(
            f"/api/settings/secrets/{secret_key}",
            headers=headers
        )
        assert response.status_code == 404

    async def test_delete_secret_not_found(self, client: TestClient, headers: dict):
        """
        Test deleting a secret that doesn't exist.
        Verifies that 404 is returned.
        """
        response = client.delete(
            "/api/settings/secrets/NONEXISTENT",
            headers=headers
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_isolation(self, client: TestClient, headers: dict, user_id: str):
        """
        Test that users can only access their own secrets.
        Verifies user isolation at the API level.
        """
        # Create a secret as user_id via PUT
        secret_key = "OPENAI_API_KEY"
        create_data = {
            "value": "user_specific_value"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=create_data, headers=headers)
        assert response.status_code == 200

        # Create a secret in the database for a different user
        await Secret.upsert(
            user_id="other_user",
            key="ANTHROPIC_API_KEY",
            value="other_value"
        )

        # Verify current user can only see their configured secret (OPENAI_API_KEY)
        # and all possible secrets from registry
        response = client.get("/api/settings/secrets", headers=headers)
        assert response.status_code == 200
        secrets = response.json()["secrets"]
        # Should have many secrets from registry, at least one configured
        assert len(secrets) > 0

        # Find the OPENAI_API_KEY and verify it's configured by current user
        openai_secret = next((s for s in secrets if s["key"] == "OPENAI_API_KEY"), None)
        assert openai_secret is not None
        assert openai_secret["is_configured"] is True
        assert openai_secret["user_id"] == user_id

        # Verify current user cannot access other user's decrypted secret
        response = client.get(
            "/api/settings/secrets/ANTHROPIC_API_KEY?decrypt=true",
            headers=headers
        )
        assert response.status_code == 404

    async def test_create_upsert_behavior(self, client: TestClient, headers: dict, user_id: str):
        """
        Test that updating a secret with same key upserts correctly.
        Verifies upsert behavior via PUT.
        """
        secret_key = "GEMINI_API_KEY"

        # Create initial secret
        create_data = {
            "value": "initial_value",
            "description": "Initial"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=create_data, headers=headers)
        assert response.status_code == 200
        first_id = response.json()["id"]

        # Update same key (should upsert)
        update_data = {
            "value": "updated_value",
            "description": "Updated"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=update_data, headers=headers)
        assert response.status_code == 200
        second_id = response.json()["id"]

        # Verify it's the same secret (same ID)
        assert first_id == second_id

        # Verify new value is stored
        response = client.get(
            f"/api/settings/secrets/{secret_key}?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == "updated_value"
        assert response.json()["description"] == "Updated"

    async def test_create_secret_special_characters(self, client: TestClient, headers: dict, user_id: str):
        """
        Test creating secrets with special characters and unicode.
        Verifies that complex values are handled correctly.
        """
        special_value = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        unicode_value = "Hello 世界 🔐 мир"

        # Create secret with special characters via PUT
        put_data = {
            "value": special_value
        }
        response = client.put("/api/settings/secrets/HF_TOKEN", json=put_data, headers=headers)
        assert response.status_code == 200

        # Retrieve and verify
        response = client.get(
            "/api/settings/secrets/HF_TOKEN?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == special_value

        # Create secret with unicode via PUT
        put_data = {
            "value": unicode_value
        }
        response = client.put("/api/settings/secrets/REPLICATE_API_TOKEN", json=put_data, headers=headers)
        assert response.status_code == 200

        # Retrieve and verify
        response = client.get(
            "/api/settings/secrets/REPLICATE_API_TOKEN?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == unicode_value

    async def test_create_secret_empty_value(self, client: TestClient, headers: dict, user_id: str):
        """
        Test creating a secret with empty value.
        Verifies that empty values are handled correctly.
        """
        put_data = {
            "value": ""
        }
        response = client.put("/api/settings/secrets/ELEVENLABS_API_KEY", json=put_data, headers=headers)
        assert response.status_code == 200

        # Retrieve and verify empty value is preserved
        response = client.get(
            "/api/settings/secrets/ELEVENLABS_API_KEY?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == ""

    async def test_create_secret_large_value(self, client: TestClient, headers: dict, user_id: str):
        """
        Test creating a secret with large value.
        Verifies that large secrets are handled correctly.
        """
        large_value = "x" * 100000  # 100KB secret
        put_data = {
            "value": large_value
        }
        response = client.put("/api/settings/secrets/FAL_API_KEY", json=put_data, headers=headers)
        assert response.status_code == 200

        # Retrieve and verify
        response = client.get(
            "/api/settings/secrets/FAL_API_KEY?decrypt=true",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["value"] == large_value
        assert len(response.json()["value"]) == 100000

    async def test_get_secret_with_false_decrypt_param(self, client: TestClient, headers: dict, user_id: str):
        """
        Test getting a secret with decrypt=false parameter.
        Verifies that decryption can be explicitly disabled.
        """
        # Create a secret via PUT
        put_data = {
            "value": "secret_value"
        }
        response = client.put("/api/settings/secrets/AIME_API_KEY", json=put_data, headers=headers)
        assert response.status_code == 200

        # Get with decrypt=false (explicit)
        response = client.get(
            "/api/settings/secrets/AIME_API_KEY?decrypt=false",
            headers=headers
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["key"] == "AIME_API_KEY"
        # Value should not be included when decrypt=false
        assert "value" not in json_response or json_response.get("value") is None

    async def test_create_secret_multiple_times_increments_timestamp(self, client: TestClient, headers: dict, user_id: str):
        """
        Test that updating a secret changes the updated_at timestamp.
        Verifies timestamp tracking.
        """
        import time

        secret_key = "SERPAPI_API_KEY"

        # Create initial secret via PUT
        put_data = {
            "value": "value1"
        }
        response = client.put(f"/api/settings/secrets/{secret_key}", json=put_data, headers=headers)
        assert response.status_code == 200
        first_updated_at = response.json()["updated_at"]

        # Wait a bit
        time.sleep(0.1)

        # Update the secret
        update_data = {
            "value": "value2"
        }
        response = client.put(
            f"/api/settings/secrets/{secret_key}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 200
        second_updated_at = response.json()["updated_at"]

        # Verify timestamp was updated
        assert second_updated_at > first_updated_at
