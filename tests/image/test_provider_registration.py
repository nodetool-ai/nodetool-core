"""Test that image providers are properly registered."""

import pytest
from nodetool.image.providers import list_image_providers


def test_providers_are_registered():
    """Test that image providers are auto-loaded and registered."""
    providers = list_image_providers()

    # Check that providers list is not empty
    assert len(providers) > 0, "No image providers registered"

    # Check for expected providers (if packages are installed)
    # These should be registered when packages are imported
    print(f"Registered image providers: {providers}")

    # At minimum, we should have providers from installed packages
    # Note: The actual providers depend on which packages are installed
    assert isinstance(providers, list)
    assert all(isinstance(p, str) for p in providers)


def test_provider_loading_is_idempotent():
    """Test that calling list_image_providers multiple times doesn't cause issues."""
    providers1 = list_image_providers()
    providers2 = list_image_providers()

    assert providers1 == providers2, "Provider list should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

