"""
DocuBot Encryption Module
Secure encryption for sensitive data storage
"""

import os
import base64
import json
import hashlib
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionManager:
    """Manage encryption and decryption of sensitive data"""
    
    def __init__(self, key_file: Optional[Path] = None):
        self.key_file = key_file or Path.home() / ".docubot" / "secret.key"
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._fernet = None
        self._load_or_generate_key()
    
    def _load_or_generate_key(self):
        """Load existing key or generate new one"""
        if self.key_file.exists() and self.key_file.stat().st_size > 0:
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        
        self._fernet = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self._fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        return self._fernet.decrypt(encrypted_data)
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt string and return base64 encoded result"""
        encrypted = self.encrypt(text.encode('utf-8'))
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt base64 encoded string"""
        encrypted_data = base64.b64decode(encrypted_text.encode('ascii'))
        decrypted = self.decrypt(encrypted_data)
        return decrypted.decode('utf-8')
    
    def encrypt_file(self, input_file: Path, output_file: Optional[Path] = None):
        """Encrypt file contents"""
        if output_file is None:
            output_file = input_file.with_suffix(input_file.suffix + '.enc')
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted = self.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_file(self, input_file: Path, output_file: Optional[Path] = None):
        """Decrypt file contents"""
        if output_file is None:
            if input_file.suffix == '.enc':
                output_file = input_file.with_suffix('')
            else:
                output_file = input_file.with_suffix(input_file.suffix + '.dec')
        
        with open(input_file, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.decrypt(encrypted)
        
        with open(output_file, 'wb') as f:
            f.write(decrypted)
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def rotate_key(self, new_key_file: Optional[Path] = None):
        """Rotate to new encryption key"""
        old_key = self._fernet._signing_key + self._fernet._encryption_key
        
        new_key = Fernet.generate_key()
        new_fernet = Fernet(new_key)
        
        self._fernet = new_fernet
        
        key_file = new_key_file or self.key_file
        with open(key_file, 'wb') as f:
            f.write(new_key)
        
        return old_key


class SensitiveDataStore:
    """Store and retrieve sensitive data with encryption"""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        self.encryption = encryption_manager or EncryptionManager()
        self.data_dir = Path.home() / ".docubot" / "secure_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def store(self, key: str, data: Union[str, dict, list], metadata: Optional[dict] = None):
        """Store sensitive data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        encrypted = self.encryption.encrypt_string(data_str)
        
        record = {
            'data': encrypted,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
    
    def retrieve(self, key: str) -> Optional[Union[str, dict, list]]:
        """Retrieve sensitive data"""
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        if not record_file.exists():
            return None
        
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        decrypted = self.encryption.decrypt_string(record['data'])
        
        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted
    
    def delete(self, key: str):
        """Delete sensitive data"""
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        if record_file.exists():
            record_file.unlink()


# Global encryption instance
_encryption_manager = None

def get_encryption_manager() -> EncryptionManager:
    """Get singleton encryption manager instance"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_sensitive_data_store() -> SensitiveDataStore:
    """Get singleton sensitive data store instance"""
    return SensitiveDataStore(get_encryption_manager())
