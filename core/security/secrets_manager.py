"""Secrets Manager Service for handling credentials securely."""

import base64
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

from core.logging import get_logger

logger = get_logger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""

    @abstractmethod
    async def store_secret(self, secret_id: str, secret_data: Dict[str, Any]) -> str:
        """Store a secret and return the URI."""
        pass

    @abstractmethod
    async def get_secret(self, secret_uri: str) -> Dict[str, Any]:
        """Retrieve a secret by URI."""
        pass

    @abstractmethod
    async def delete_secret(self, secret_uri: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    async def update_secret(self, secret_uri: str, secret_data: Dict[str, Any]) -> str:
        """Update an existing secret."""
        pass


class LocalSecretsProvider(SecretsProvider):
    """Local secrets provider using encryption for development/testing
    In production, use AWS Secrets Manager, HashiCorp Vault, etc.
    .
    """

    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            # Generate a key from environment variable or create a new one
            key = os.getenv("SECRETS_ENCRYPTION_KEY")
            if not key:
                key = Fernet.generate_key()
                logger.warning("Using generated encryption key. Set SECRETS_ENCRYPTION_KEY env var in production")
            if isinstance(key, bytes):
                self.cipher = Fernet(key)
            elif isinstance(key, str):
                self.cipher = Fernet(key.encode())
            else:
                self.cipher = Fernet(bytes(key))

        # In a real implementation, this would be a database or external storage
        self._storage: Dict[str, bytes] = {}

    async def store_secret(self, secret_id: str, secret_data: Dict[str, Any]) -> str:
        """Store encrypted secret locally."""
        # Convert secret data to JSON
        secret_json = json.dumps(secret_data)

        # Encrypt the data
        encrypted_data = self.cipher.encrypt(secret_json.encode())

        # Store with a unique ID
        storage_key = f"{secret_id}_{uuid.uuid4()}"
        self._storage[storage_key] = encrypted_data

        # Return a URI that identifies this secret
        return f"local://{storage_key}"

    async def get_secret(self, secret_uri: str) -> Dict[str, Any]:
        """Retrieve and decrypt secret."""
        if not secret_uri.startswith("local://"):
            raise ValueError(f"Invalid secret URI format: {secret_uri}")

        storage_key = secret_uri.replace("local://", "")

        if storage_key not in self._storage:
            raise KeyError(f"Secret not found: {secret_uri}")

        # Decrypt the data
        encrypted_data = self._storage[storage_key]
        decrypted_json = self.cipher.decrypt(encrypted_data).decode()

        return json.loads(decrypted_json)

    async def delete_secret(self, secret_uri: str) -> bool:
        """Delete a secret."""
        if not secret_uri.startswith("local://"):
            return False

        storage_key = secret_uri.replace("local://", "")

        if storage_key in self._storage:
            del self._storage[storage_key]
            return True

        return False

    async def update_secret(self, secret_uri: str, secret_data: Dict[str, Any]) -> str:
        """Update an existing secret."""
        if not secret_uri.startswith("local://"):
            raise ValueError(f"Invalid secret URI format: {secret_uri}")

        storage_key = secret_uri.replace("local://", "")

        if storage_key not in self._storage:
            raise KeyError(f"Secret not found: {secret_uri}")

        # Encrypt the new data
        secret_json = json.dumps(secret_data)
        encrypted_data = self.cipher.encrypt(secret_json.encode())

        # Update the storage
        self._storage[storage_key] = encrypted_data

        return secret_uri


class AWSSecretsProvider(SecretsProvider):
    """AWS Secrets Manager provider for production use."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        # Initialize boto3 client here
        # self.client = boto3.client('secretsmanager', region_name=region)

    async def store_secret(self, secret_id: str, secret_data: Dict[str, Any]) -> str:
        """Store secret in AWS Secrets Manager."""
        # Implementation would use boto3 to store in AWS
        # response = self.client.create_secret(
        #     Name=secret_id,
        #     SecretString=json.dumps(secret_data)
        # )
        # return f"aws-sm://{self.region}/{secret_id}"
        raise NotImplementedError("AWS Secrets Manager provider not implemented")

    async def get_secret(self, secret_uri: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager."""
        # Implementation would use boto3 to retrieve from AWS
        raise NotImplementedError("AWS Secrets Manager provider not implemented")

    async def delete_secret(self, secret_uri: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        # Implementation would use boto3 to delete from AWS
        raise NotImplementedError("AWS Secrets Manager provider not implemented")

    async def update_secret(self, secret_uri: str, secret_data: Dict[str, Any]) -> str:
        """Update secret in AWS Secrets Manager."""
        # Implementation would use boto3 to update in AWS
        raise NotImplementedError("AWS Secrets Manager provider not implemented")


class SecretsManager:
    """Main secrets manager that delegates to appropriate provider."""

    def __init__(self, provider: Optional[SecretsProvider] = None):
        self.provider = provider or LocalSecretsProvider()
        # Initialize cipher for field-level encryption
        if isinstance(self.provider, LocalSecretsProvider):
            self.cipher = self.provider.cipher
        else:
            # For other providers, create a local cipher for field encryption
            key = os.getenv("SECRETS_ENCRYPTION_KEY")
            if not key:
                key = Fernet.generate_key()
            if isinstance(key, str):
                key_bytes = key.encode()
            elif isinstance(key, (bytes, bytearray)):
                key_bytes = bytes(key)
            elif isinstance(key, memoryview):
                key_bytes = key.tobytes()
            else:
                raise TypeError("SECRETS_ENCRYPTION_KEY must be str, bytes, bytearray, or memoryview")
            self.cipher = Fernet(key_bytes)

    def encrypt_value(self, value: Any) -> str | None:
        """Encrypt a single value (field-level encryption).

        Args:
            value: The value to encrypt (will be JSON serialized if not string)

        Returns:
            Base64 encoded encrypted string with prefix.

        """
        if value is None:
            return None

        # Convert to string if needed
        if not isinstance(value, str):
            value_str = json.dumps(value)
        else:
            value_str = value

        # Encrypt and encode
        encrypted = self.cipher.encrypt(value_str.encode())
        # Add prefix to identify encrypted values
        return f"__encrypted__:{base64.b64encode(encrypted).decode()}"

    def decrypt_value(self, encrypted_value: str) -> Any:
        """Decrypt a single encrypted value.

        Args:
            encrypted_value: The encrypted value with prefix

        Returns:
            The decrypted value (JSON parsed if applicable)

        .

        """
        if not encrypted_value or not isinstance(encrypted_value, str):
            return encrypted_value

        # Check for encryption prefix
        if not encrypted_value.startswith("__encrypted__:"):
            return encrypted_value

        try:
            # Remove prefix and decode
            encrypted_data = encrypted_value.replace("__encrypted__:", "")
            encrypted_bytes = base64.b64decode(encrypted_data)

            # Decrypt
            decrypted_str = self.cipher.decrypt(encrypted_bytes).decode()

            # Try to parse as JSON if it looks like JSON
            if decrypted_str.startswith(("[", "{", '"')) or decrypted_str in (
                "true",
                "false",
                "null",
            ):
                try:
                    return json.loads(decrypted_str)
                except json.JSONDecodeError:
                    return decrypted_str

            return decrypted_str
        except Exception as e:
            # Enhanced error handling with more specific error types
            error_msg = str(e).lower()
            if "invalid token" in error_msg or "decrypt" in error_msg:
                logger.warning(f"Warning: Failed to decrypt value - likely encrypted with different key. Error: {e}")
                raise ValueError(
                    f"Invalid encryption token - data may have been encrypted with a different key. Original error: {e}"
                ) from e
            else:
                logger.error(f"Warning: Failed to decrypt value due to unexpected error: {e}")
                raise ValueError(f"Decryption failed: {e}") from e

    def encrypt_config_credentials(self, config: Dict[str, Any], credential_keys: List[str]) -> Dict[str, Any]:
        """Encrypt specific credential fields within a config dictionary.

        Args:
            config: The configuration dictionary
            credential_keys: List of keys that contain sensitive data to encrypt

        Returns:
            Config dictionary with encrypted credential fields

        .

        """
        encrypted_config = config.copy()

        for key in credential_keys:
            if key in encrypted_config and encrypted_config[key] is not None:
                encrypted_config[key] = self.encrypt_value(encrypted_config[key])

        return encrypted_config

    def decrypt_config_credentials(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt any encrypted fields in a config dictionary.

        Args:
            config: The configuration dictionary with potentially encrypted fields

        Returns:
            Config dictionary with decrypted fields

        """
        decrypted_config = {}
        decryption_errors = []

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("__encrypted__:"):
                try:
                    decrypted_config[key] = self.decrypt_value(value)
                except ValueError as e:
                    # Log the error but continue with other fields
                    decryption_errors.append(f"Failed to decrypt field '{key}': {e}")
                    # Keep the encrypted value or set to None
                    decrypted_config[key] = None
            elif isinstance(value, dict):
                # Recursively decrypt nested dictionaries
                try:
                    decrypted_config[key] = self.decrypt_config_credentials(value)
                except Exception as e:
                    decryption_errors.append(f"Failed to decrypt nested object '{key}': {e}")
                    decrypted_config[key] = value
            else:
                decrypted_config[key] = value

        # If there were decryption errors, log them but don't fail the entire operation
        if decryption_errors:
            logger.warning(f"Decryption warnings: {'; '.join(decryption_errors)}")

        return decrypted_config

    async def store_credentials(self, data_source_id: str, credentials: Dict[str, Any]) -> str:
        """Store credentials for a data source.

        Args:
            data_source_id: The data source identifier
            credentials: Dictionary containing sensitive credentials

        Returns:
            Secret URI that can be stored in the database

        """
        secret_id = f"datasource/{data_source_id}"
        return await self.provider.store_secret(secret_id, credentials)

    async def get_credentials(self, secret_uri: str) -> Dict[str, Any]:
        """Retrieve credentials using the secret URI.

        Args:
            secret_uri: The URI returned when storing the secret

        Returns:
            Dictionary containing the credentials

        """
        return await self.provider.get_secret(secret_uri)

    async def update_credentials(self, secret_uri: str, credentials: Dict[str, Any]) -> str:
        """Update existing credentials.

        Args:
            secret_uri: The existing secret URI
            credentials: New credentials to store

        Returns:
            Updated secret URI (may be the same)

        """
        return await self.provider.update_secret(secret_uri, credentials)

    async def delete_credentials(self, secret_uri: str) -> bool:
        """Delete credentials.

        Args:
            secret_uri: The secret URI to delete

        Returns:
            True if deleted successfully

        """
        return await self.provider.delete_secret(secret_uri)


# Global instance (in production, initialize with appropriate provider)
secrets_manager = SecretsManager()
