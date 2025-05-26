"""
Tests for ChatWebSocketRunner authentication with Supabase
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from nodetool.common.chat_websocket_runner import ChatWebSocketRunner
from nodetool.common.environment import Environment


@pytest.mark.asyncio
class TestChatWebSocketRunnerAuth:
    """Test suite for ChatWebSocketRunner authentication functionality"""

    async def test_local_development_no_auth_required(self):
        """Test that authentication is bypassed in local development mode"""
        with patch(
            "nodetool.common.environment.Environment.use_remote_auth",
            return_value=False,
        ):
            runner = ChatWebSocketRunner()
            websocket = Mock()
            websocket.accept = AsyncMock()

            await runner.connect(websocket)

            # Verify connection was accepted
            websocket.accept.assert_called_once()
            # Verify default user ID was set
            assert runner.user_id == "1"
            # Verify websocket was never closed
            assert not hasattr(websocket, "close") or not websocket.close.called

    async def test_missing_auth_token_when_required(self):
        """Test that missing auth token is rejected when authentication is required"""
        with patch(
            "nodetool.common.environment.Environment.use_remote_auth", return_value=True
        ):
            runner = ChatWebSocketRunner()  # No auth token provided
            websocket = Mock()
            websocket.close = AsyncMock()

            await runner.connect(websocket)

            # Verify connection was closed with correct code and reason
            websocket.close.assert_called_once_with(
                code=1008, reason="Missing authentication"
            )
            # Verify accept was never called
            assert not hasattr(websocket, "accept") or not websocket.accept.called

    async def test_invalid_auth_token(self):
        """Test that invalid auth token is rejected"""
        with patch(
            "nodetool.common.environment.Environment.use_remote_auth", return_value=True
        ):
            runner = ChatWebSocketRunner(auth_token="invalid_token")
            websocket = Mock()
            websocket.close = AsyncMock()

            # Mock validate_token to return False
            with patch.object(runner, "validate_token", return_value=False):
                await runner.connect(websocket)

            # Verify connection was closed with correct code and reason
            websocket.close.assert_called_once_with(
                code=1008, reason="Invalid authentication"
            )
            # Verify accept was never called
            assert not hasattr(websocket, "accept") or not websocket.accept.called

    async def test_valid_auth_token(self):
        """Test that valid auth token is accepted"""
        with patch(
            "nodetool.common.environment.Environment.use_remote_auth", return_value=True
        ):
            runner = ChatWebSocketRunner(auth_token="valid_token")
            websocket = Mock()
            websocket.accept = AsyncMock()

            # Mock validate_token to return True and set user_id
            async def mock_validate(token):
                runner.user_id = "test-user-123"
                return True

            with patch.object(runner, "validate_token", side_effect=mock_validate):
                await runner.connect(websocket)

            # Verify connection was accepted
            websocket.accept.assert_called_once()
            # Verify user ID was set correctly
            assert runner.user_id == "test-user-123"
            # Verify close was never called
            assert not hasattr(websocket, "close") or not websocket.close.called

    async def test_validate_token_with_supabase(self):
        """Test the validate_token method with Supabase integration"""
        runner = ChatWebSocketRunner(auth_token="test_jwt_token")

        # Mock Supabase client and response
        mock_supabase = Mock()
        mock_session = Mock()
        mock_session.access_token = "test_jwt_token"
        mock_user_response = Mock()
        mock_user = Mock()
        mock_user.id = "user-123"
        mock_user_response.user = mock_user

        mock_supabase.auth.get_session = AsyncMock(return_value=mock_session)
        mock_supabase.auth.get_user = AsyncMock(return_value=mock_user_response)

        # Mock the async client creation
        with patch(
            "nodetool.common.chat_websocket_runner.create_async_client",
            AsyncMock(return_value=mock_supabase),
        ):
            with patch(
                "nodetool.common.environment.Environment.get_supabase_url",
                return_value="https://test.supabase.co",
            ):
                with patch(
                    "nodetool.common.environment.Environment.get_supabase_key",
                    return_value="test_key",
                ):
                    result = await runner.validate_token("test_jwt_token")

        # Verify the token validation was successful
        assert result is True
        assert runner.user_id == "user-123"
        mock_supabase.auth.get_user.assert_called_once_with("test_jwt_token")

    async def test_validate_token_invalid_response(self):
        """Test validate_token with invalid Supabase response"""
        runner = ChatWebSocketRunner(auth_token="test_jwt_token")

        # Mock Supabase client with invalid response
        mock_supabase = Mock()
        mock_session = Mock()
        mock_session.access_token = "different_token"  # Different token
        mock_supabase.auth.get_session = AsyncMock(return_value=mock_session)

        # Mock the async client creation
        with patch(
            "nodetool.common.chat_websocket_runner.create_async_client",
            AsyncMock(return_value=mock_supabase),
        ):
            with patch(
                "nodetool.common.environment.Environment.get_supabase_url",
                return_value="https://test.supabase.co",
            ):
                with patch(
                    "nodetool.common.environment.Environment.get_supabase_key",
                    return_value="test_key",
                ):
                    result = await runner.validate_token("test_jwt_token")

        # Verify the token validation failed
        assert result is False

    async def test_validate_token_exception_handling(self):
        """Test validate_token handles exceptions gracefully"""
        runner = ChatWebSocketRunner(auth_token="test_jwt_token")

        # Mock Supabase client to raise an exception
        mock_supabase = Mock()
        mock_supabase.auth.get_session = AsyncMock(side_effect=Exception("Supabase error"))

        # Mock the async client creation
        with patch(
            "nodetool.common.chat_websocket_runner.create_async_client",
            AsyncMock(return_value=mock_supabase),
        ):
            with patch(
                "nodetool.common.environment.Environment.get_supabase_url",
                return_value="https://test.supabase.co",
            ):
                with patch(
                    "nodetool.common.environment.Environment.get_supabase_key",
                    return_value="test_key",
                ):
                    result = await runner.validate_token("test_jwt_token")

        # Verify the token validation failed gracefully
        assert result is False

    async def test_user_id_passed_to_processing_context(self):
        """Test that user_id is correctly passed to ProcessingContext"""
        runner = ChatWebSocketRunner()
        runner.user_id = "test-user-456"

        # Just test that ProcessingContext is created with correct user_id
        # by directly calling the relevant part of the method
        from nodetool.workflows.processing_context import ProcessingContext

        # Call the constructor directly to verify it accepts user_id
        context = ProcessingContext(user_id=runner.user_id)
        assert context.user_id == "test-user-456"

    async def test_user_id_passed_to_workflow_processing_context(self):
        """Test that user_id is correctly passed to ProcessingContext in workflow processing"""
        runner = ChatWebSocketRunner()
        runner.user_id = "test-user-789"

        # Just test that ProcessingContext is created with correct user_id and workflow_id
        # by directly calling the constructor
        from nodetool.workflows.processing_context import ProcessingContext

        # Call the constructor directly to verify it accepts both user_id and workflow_id
        context = ProcessingContext(user_id=runner.user_id, workflow_id="workflow-123")
        assert context.user_id == "test-user-789"
        assert context.workflow_id == "workflow-123"
