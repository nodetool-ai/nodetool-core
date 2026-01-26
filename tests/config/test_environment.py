"""Tests for environment configuration security features."""

from unittest.mock import patch

from nodetool.config.environment import Environment


class TestInsecureAuthBinding:
    """Test the insecure authentication binding detection."""

    def test_no_warning_with_localhost(self):
        """No warning when binding to localhost with local auth."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="local"):
            warnings = Environment.check_insecure_auth_binding("127.0.0.1")
            assert len(warnings) == 0

    def test_no_warning_with_static_auth(self):
        """No warning when using static auth even with network binding."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="static"):
            warnings = Environment.check_insecure_auth_binding("0.0.0.0")
            assert len(warnings) == 0

    def test_no_warning_with_supabase_auth(self):
        """No warning when using supabase auth even with network binding."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="supabase"):
            warnings = Environment.check_insecure_auth_binding("0.0.0.0")
            assert len(warnings) == 0

    def test_warning_with_local_auth_network_binding(self):
        """Warning when using local auth with network binding."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="local"):
            warnings = Environment.check_insecure_auth_binding("0.0.0.0")
            assert len(warnings) > 0
            assert any("SECURITY WARNING" in w for w in warnings)
            assert any("AUTH_PROVIDER" in w for w in warnings)

    def test_warning_with_none_auth_network_binding(self):
        """Warning when using none auth with network binding."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="none"):
            warnings = Environment.check_insecure_auth_binding("0.0.0.0")
            assert len(warnings) > 0
            assert any("SECURITY WARNING" in w for w in warnings)

    def test_warning_with_ipv6_any_binding(self):
        """Warning when binding to IPv6 any address (::) with local auth."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="local"):
            warnings = Environment.check_insecure_auth_binding("::")
            assert len(warnings) > 0
            assert any("SECURITY WARNING" in w for w in warnings)

    def test_warning_with_empty_host_binding(self):
        """Warning when binding to empty host (all interfaces) with local auth."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="local"):
            warnings = Environment.check_insecure_auth_binding("")
            assert len(warnings) > 0
            assert any("SECURITY WARNING" in w for w in warnings)

    def test_warning_includes_remediation_advice(self):
        """Warning includes advice on how to fix the issue."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="none"):
            warnings = Environment.check_insecure_auth_binding("0.0.0.0")
            # Check that remediation advice is included
            warning_text = " ".join(warnings)
            assert "static" in warning_text.lower() or "supabase" in warning_text.lower()

    def test_no_warning_for_specific_interface(self):
        """No warning when binding to specific interface IP."""
        with patch.object(Environment, "get_auth_provider_kind", return_value="local"):
            warnings = Environment.check_insecure_auth_binding("192.168.1.100")
            assert len(warnings) == 0
