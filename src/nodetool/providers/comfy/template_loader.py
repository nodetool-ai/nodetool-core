"""
Template loader for ComfyUI YAML mappings.

Provides functionality to load, validate, and list YAML-based template mappings
that define how to execute ComfyUI workflows.
"""

import os
from pathlib import Path

import yaml

from nodetool.config.logging_config import get_logger
from nodetool.providers.comfy.template_models import TemplateInfo, TemplateMapping

log = get_logger(__name__)


def get_default_templates_dir() -> Path:
    """Get the default templates directory path.

    Returns the path to the templates directory within the comfy package,
    which can be overridden via the COMFY_TEMPLATE_DIR environment variable.
    """
    env_dir = os.environ.get("COMFY_TEMPLATE_DIR")
    if env_dir:
        return Path(env_dir)

    # Default to the templates directory within this package
    return Path(__file__).parent / "templates"


class TemplateLoader:
    """Loader for ComfyUI YAML template mappings.

    Handles loading, validation, and discovery of template mapping files
    that define how to execute ComfyUI workflows.
    """

    def __init__(self, templates_dir: str | Path | None = None):
        """Initialize the template loader.

        Args:
            templates_dir: Path to the directory containing YAML template mappings.
                          Defaults to the package templates directory or the value
                          of COMFY_TEMPLATE_DIR environment variable.
        """
        if templates_dir is None:
            self.templates_dir = get_default_templates_dir()
        else:
            self.templates_dir = Path(templates_dir)

    def load(self, template_id: str) -> TemplateMapping | None:
        """Load a template mapping by ID.

        Args:
            template_id: The unique identifier of the template (filename without .yaml)

        Returns:
            TemplateMapping if found and valid, None otherwise
        """
        yaml_path = self.templates_dir / f"{template_id}.yaml"

        if not yaml_path.exists():
            log.warning("Template mapping not found: %s", yaml_path)
            return None

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            mapping = TemplateMapping.model_validate(data)
            log.debug("Loaded template mapping: %s", template_id)
            return mapping

        except yaml.YAMLError as e:
            log.error("Failed to parse YAML for template %s: %s", template_id, e)
            return None
        except Exception as e:
            log.error("Failed to load template %s: %s", template_id, e)
            return None

    def list_templates(
        self,
        template_types: list[str] | None = None,
    ) -> list[TemplateInfo]:
        """List all available templates, optionally filtered by types.

        Args:
            template_types: Filter by template types (e.g., ["text_to_image", "image_to_image"]).
                           If None, returns all templates.

        Returns:
            List of TemplateInfo for user selection.
        """
        templates: list[TemplateInfo] = []

        if not self.templates_dir.exists():
            log.warning("Templates directory does not exist: %s", self.templates_dir)
            return templates

        for yaml_file in self.templates_dir.glob("*.yaml"):
            template_id = yaml_file.stem
            mapping = self.load(template_id)

            if mapping is None:
                continue

            if template_types is None or mapping.template_type in template_types:
                templates.append(
                    TemplateInfo(
                        template_id=mapping.template_id,
                        template_name=mapping.template_name,
                        template_type=mapping.template_type,
                        description=mapping.description,
                        inputs=list(mapping.inputs.keys()),
                        outputs=list(mapping.outputs.keys()),
                    )
                )

        return templates

    def get_template_info(self, template_id: str) -> TemplateInfo | None:
        """Get info for a single template.

        Args:
            template_id: The unique identifier of the template

        Returns:
            TemplateInfo if found and valid, None otherwise
        """
        mapping = self.load(template_id)
        if mapping is None:
            return None

        return TemplateInfo(
            template_id=mapping.template_id,
            template_name=mapping.template_name,
            template_type=mapping.template_type,
            description=mapping.description,
            inputs=list(mapping.inputs.keys()),
            outputs=list(mapping.outputs.keys()),
        )

    def get_image_templates(self) -> list[TemplateInfo]:
        """Get all image generation templates (text_to_image and image_to_image).

        Returns:
            List of TemplateInfo for image generation templates.
        """
        return self.list_templates(template_types=["text_to_image", "image_to_image"])

    def get_video_templates(self) -> list[TemplateInfo]:
        """Get all video generation templates (image_to_video and text_to_video).

        Returns:
            List of TemplateInfo for video generation templates.
        """
        return self.list_templates(template_types=["image_to_video", "text_to_video"])
