"""
Role-Based Access Control (RBAC) for document retrieval filtering.
"""

from typing import List, Dict, Any


class RBACFilter:
    """
    Enforces role-based access control at the retrieval stage.

    Role Hierarchy (higher roles inherit lower-role access):
        admin > researcher > analyst > viewer

    Each document chunk has a 'role_access' metadata field listing
    which roles can access it.
    """

    ROLE_HIERARCHY = {
        "admin": 4,
        "researcher": 3,
        "analyst": 2,
        "viewer": 1,
    }

    @classmethod
    def get_accessible_roles(cls, user_role: str) -> List[str]:
        """
        Get all roles a user can access (their role + all lower roles).

        Example: 'researcher' can access documents tagged for
        'researcher', 'analyst', and 'viewer'.
        """
        user_level = cls.ROLE_HIERARCHY.get(user_role, 0)
        accessible = [
            role for role, level in cls.ROLE_HIERARCHY.items() if level <= user_level
        ]
        return accessible

    @classmethod
    def can_access(cls, user_role: str, doc_roles: List[str]) -> bool:
        """
        Check if a user role can access a document.

        A user can access a document if their role (or any lower role)
        is in the document's role_access list.
        """
        accessible = cls.get_accessible_roles(user_role)
        return any(role in doc_roles for role in accessible)

    @classmethod
    def get_role_filter(cls, user_role: str) -> List[str]:
        """
        Get the list of roles to filter by in retrieval queries.
        This is used as a metadata filter in ChromaDB and BM25.
        """
        return cls.get_accessible_roles(user_role)

    @classmethod
    def filter_results(
        cls,
        results: List[Dict[str, Any]],
        user_role: str,
    ) -> List[Dict[str, Any]]:
        """
        Post-retrieval filter to ensure RBAC compliance.
        Acts as a safety net in case the query-time filter misses anything.
        """
        accessible = cls.get_accessible_roles(user_role)
        filtered = []

        for result in results:
            metadata = result.get("metadata", {})
            doc_roles_str = metadata.get("role_access", "viewer")

            # Handle both string and list formats
            if isinstance(doc_roles_str, str):
                doc_roles = [r.strip() for r in doc_roles_str.split(",")]
            else:
                doc_roles = doc_roles_str

            if any(role in doc_roles for role in accessible):
                filtered.append(result)

        return filtered
