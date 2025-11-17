from typing import List

from pydantic import BaseModel, Field


class PackageInfo(BaseModel):
    """
    Package information model for nodetool.
    This is the model for the package index in the registry.
    """

    name: str
    description: str
    repo_id: str = Field(description="Repository ID in the format <owner>/<project>")
    namespaces: List[str] = Field(
        default_factory=list, description="Namespaces provided by this package"
    )


class AssetInfo(BaseModel):
    """
    Asset information model for nodetool packages.
    Represents files provided by packages in their assets directories.
    """

    package_name: str = Field(description="Name of the package providing the asset")
    name: str = Field(description="Asset file name")
    path: str = Field(description="Full path to the asset file")
