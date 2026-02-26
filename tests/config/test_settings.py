"""Tests for settings loading and saving."""

import os
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

from nodetool.config.settings import load_settings, save_settings


def test_load_settings_file_exists_and_valid(tmp_path):
    """Test loading settings from a valid YAML file."""
    # Create a temporary settings file
    settings_file = tmp_path / "settings.yaml"
    settings_data = {"key": "value", "nested": {"foo": "bar"}}
    with open(settings_file, "w") as f:
        yaml.dump(settings_data, f)

    # Mock get_system_file_path to return our temporary file
    with patch("nodetool.config.settings.get_system_file_path", return_value=settings_file):
        loaded_settings = load_settings()
        assert loaded_settings == settings_data


def test_load_settings_file_does_not_exist(tmp_path):
    """Test loading settings when the file does not exist."""
    # Create a path that doesn't exist
    settings_file = tmp_path / "nonexistent_settings.yaml"

    # Mock get_system_file_path to return our nonexistent file path
    with patch("nodetool.config.settings.get_system_file_path", return_value=settings_file):
        loaded_settings = load_settings()
        assert loaded_settings == {}


def test_load_settings_empty_file(tmp_path):
    """Test loading settings from an empty file."""
    # Create an empty settings file
    settings_file = tmp_path / "empty_settings.yaml"
    settings_file.touch()

    # Mock get_system_file_path to return our temporary file
    with patch("nodetool.config.settings.get_system_file_path", return_value=settings_file):
        loaded_settings = load_settings()
        assert loaded_settings == {}


def test_save_settings(tmp_path):
    """Test saving settings to a YAML file."""
    # Define a target settings file path
    settings_file = tmp_path / "saved_settings.yaml"
    settings_data = {"new_key": "new_value", "list": [1, 2, 3]}

    # Mock get_system_file_path to return our target file path
    with patch("nodetool.config.settings.get_system_file_path", return_value=settings_file):
        save_settings(settings_data)

        # Verify the file was created and contains the correct data
        assert settings_file.exists()
        with open(settings_file, "r") as f:
            saved_data = yaml.safe_load(f)
        assert saved_data == settings_data
