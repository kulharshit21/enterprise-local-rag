"""
Simple JSON-based user store for demo purposes.
Pre-seeded with users of different roles.
"""

import os
import json
from typing import Optional, Dict, List

from security.auth import UserModel, hash_password
from config import settings


class UsersDB:
    """
    JSON file-based user database.

    In production, replace with PostgreSQL/MongoDB.
    Provides user CRUD operations and pre-seeded demo accounts.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.USERS_DB_PATH
        self._users: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load users from JSON file or initialize with defaults."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self._users = json.load(f)
        else:
            self._seed_defaults()

    def _save(self):
        """Persist users to JSON file."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self._users, f, indent=2)

    def _seed_defaults(self):
        """Create default demo users with different roles."""
        defaults = [
            {"username": "admin", "password": "admin123", "role": "admin", "email": "admin@rag.com"},
            {"username": "researcher", "password": "research123", "role": "researcher", "email": "researcher@rag.com"},
            {"username": "analyst", "password": "analyst123", "role": "analyst", "email": "analyst@rag.com"},
            {"username": "viewer", "password": "viewer123", "role": "viewer", "email": "viewer@rag.com"},
        ]

        for user in defaults:
            self._users[user["username"]] = {
                "username": user["username"],
                "hashed_password": hash_password(user["password"]),
                "role": user["role"],
                "email": user["email"],
                "disabled": False,
            }

        self._save()

    def get_user(self, username: str) -> Optional[UserModel]:
        """Get a user by username."""
        user_data = self._users.get(username)
        if user_data:
            return UserModel(**user_data)
        return None

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "viewer",
        email: str = "",
    ) -> Optional[UserModel]:
        """Create a new user. Returns None if username already exists."""
        if username in self._users:
            return None

        user_data = {
            "username": username,
            "hashed_password": hash_password(password),
            "role": role,
            "email": email,
            "disabled": False,
        }

        self._users[username] = user_data
        self._save()
        return UserModel(**user_data)

    def list_users(self) -> List[Dict]:
        """List all users (without password hashes)."""
        return [
            {"username": u["username"], "role": u["role"], "email": u["email"]}
            for u in self._users.values()
        ]

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns True if deleted."""
        if username in self._users:
            del self._users[username]
            self._save()
            return True
        return False
