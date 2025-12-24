#!/usr/bin/env python3
"""
Utility script for managing master keys in AWS Secrets Manager.

This script provides commands to:
- Store the current master key to AWS Secrets Manager
- Retrieve a master key from AWS Secrets Manager
- Generate a new master key and store it in AWS
- Delete a master key from AWS Secrets Manager

Usage:
    python -m nodetool.security.aws_secrets_util store --secret-name nodetool-master-key
    python -m nodetool.security.aws_secrets_util retrieve --secret-name nodetool-master-key
    python -m nodetool.security.aws_secrets_util generate --secret-name nodetool-master-key
    python -m nodetool.security.aws_secrets_util delete --secret-name nodetool-master-key
"""

import argparse
import os
import sys
from typing import Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class AWSSecretsUtil:
    """Utility for managing master keys in AWS Secrets Manager."""

    @staticmethod
    def get_aws_client(region: Optional[str] = None):
        """
        Create and return an AWS Secrets Manager client.

        Args:
            region: AWS region name. If not provided, uses AWS_REGION env var or defaults to us-east-1.

        Returns:
            boto3 Secrets Manager client.
        """
        try:
            import boto3
        except ImportError:
            log.error("boto3 is required. Install with: pip install boto3")
            sys.exit(1)

        region = region or os.environ.get("AWS_REGION", "us-east-1")
        session = boto3.session.Session()
        return session.client(service_name="secretsmanager", region_name=region)

    @staticmethod
    def store_master_key(secret_name: str, master_key: str, region: Optional[str] = None) -> bool:
        """
        Store a master key in AWS Secrets Manager.

        Args:
            secret_name: The name for the secret in AWS Secrets Manager.
            master_key: The master key to store.
            region: AWS region (optional).

        Returns:
            True if successful, False otherwise.
        """
        try:
            from botocore.exceptions import ClientError

            client = AWSSecretsUtil.get_aws_client(region)

            # Try to create the secret
            try:
                client.create_secret(
                    Name=secret_name, SecretString=master_key, Description="NodeTool master encryption key for secrets"
                )
                log.info(f"Master key successfully stored in AWS Secrets Manager: {secret_name}")
                return True

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceExistsException":
                    # Secret already exists, update it
                    client.put_secret_value(SecretId=secret_name, SecretString=master_key)
                    log.info(f"Master key updated in AWS Secrets Manager: {secret_name}")
                    return True
                else:
                    log.error(f"Error storing master key: {e}")
                    return False

        except Exception as e:
            log.error(f"Failed to store master key in AWS Secrets Manager: {e}")
            return False

    @staticmethod
    def retrieve_master_key(secret_name: str, region: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a master key from AWS Secrets Manager.

        Args:
            secret_name: The name of the secret in AWS Secrets Manager.
            region: AWS region (optional).

        Returns:
            The master key if found, None otherwise.
        """
        try:
            from botocore.exceptions import ClientError

            client = AWSSecretsUtil.get_aws_client(region)

            response = client.get_secret_value(SecretId=secret_name)

            # Secret can be either a string or binary
            if "SecretString" in response:
                return response["SecretString"]
            else:
                import base64

                return base64.b64decode(response["SecretBinary"]).decode()

        except Exception as e:
            log.error(f"Failed to retrieve master key: {e}")
            return None

    @staticmethod
    def delete_master_key(secret_name: str, region: Optional[str] = None, force: bool = False) -> bool:
        """
        Delete a master key from AWS Secrets Manager.

        Args:
            secret_name: The name of the secret to delete.
            region: AWS region (optional).
            force: If True, delete immediately without recovery window.

        Returns:
            True if successful, False otherwise.
        """
        try:
            client = AWSSecretsUtil.get_aws_client(region)

            kwargs = {"SecretId": secret_name}
            if force:
                kwargs["ForceDeleteWithoutRecovery"] = True
            else:
                kwargs["RecoveryWindowInDays"] = 30

            client.delete_secret(**kwargs)

            if force:
                log.warning(f"Master key permanently deleted from AWS: {secret_name}")
            else:
                log.warning(f"Master key scheduled for deletion (30 day recovery window): {secret_name}")

            return True

        except Exception as e:
            log.error(f"Failed to delete master key: {e}")
            return False

    @staticmethod
    def generate_and_store(secret_name: str, region: Optional[str] = None) -> Optional[str]:
        """
        Generate a new master key and store it in AWS Secrets Manager.

        Args:
            secret_name: The name for the secret in AWS Secrets Manager.
            region: AWS region (optional).

        Returns:
            The generated master key if successful, None otherwise.
        """
        from nodetool.security.crypto import SecretCrypto

        master_key = SecretCrypto.generate_master_key()

        if AWSSecretsUtil.store_master_key(secret_name, master_key, region):
            return master_key
        else:
            return None


def main():
    """CLI entry point for AWS Secrets Manager utilities."""
    parser = argparse.ArgumentParser(description="Manage NodeTool master keys in AWS Secrets Manager")
    parser.add_argument("--region", help="AWS region (defaults to AWS_REGION env var or us-east-1)")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store current master key to AWS")
    store_parser.add_argument("--secret-name", required=True, help="Name for the secret in AWS Secrets Manager")
    store_parser.add_argument("--key", help="Master key to store (if not provided, uses current key from keychain)")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve master key from AWS")
    retrieve_parser.add_argument("--secret-name", required=True, help="Name of the secret in AWS Secrets Manager")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate new master key and store in AWS")
    generate_parser.add_argument("--secret-name", required=True, help="Name for the secret in AWS Secrets Manager")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete master key from AWS")
    delete_parser.add_argument("--secret-name", required=True, help="Name of the secret to delete")
    delete_parser.add_argument("--force", action="store_true", help="Permanently delete without 30-day recovery window")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    util = AWSSecretsUtil()

    if args.command == "store":
        if args.key:
            master_key = args.key
        else:
            from nodetool.security.master_key import MasterKeyManager

            master_key = MasterKeyManager.export_master_key()

        success = util.store_master_key(args.secret_name, master_key, args.region)
        sys.exit(0 if success else 1)

    elif args.command == "retrieve":
        key = util.retrieve_master_key(args.secret_name, args.region)
        if key:
            print(f"Master key: {key}")
            print("\nTo use this key, set the environment variable:")
            print(f"export AWS_SECRETS_MASTER_KEY_NAME={args.secret_name}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "generate":
        key = util.generate_and_store(args.secret_name, args.region)
        if key:
            print(f"Generated and stored master key: {key}")
            print("\nIMPORTANT: Save this key securely as a backup!")
            print("\nTo use this key, set the environment variable:")
            print(f"export AWS_SECRETS_MASTER_KEY_NAME={args.secret_name}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "delete":
        success = util.delete_master_key(args.secret_name, args.region, args.force)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
