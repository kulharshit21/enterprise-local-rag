"""
Simple CLI to test the RAG system end-to-end.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def main():
    print(f"Checking system health at {BASE_URL}/health ...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        print(f"✅ System Healthy: {json.dumps(resp.json(), indent=2)}\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Is the server running? Run: uvicorn api.main:app --reload")
        sys.exit(1)

    # 1. Register/Login
    username = "test_user_1"
    password = "secure_password_123"

    print(f"Registering user '{username}'...")
    try:
        resp = requests.post(
            f"{BASE_URL}/auth/register",
            json={
                "username": username,
                "password": password,
                "role": "viewer",
                "email": "test@example.com",
            },
        )
        if resp.status_code == 400 and "already exists" in resp.text:
            print("User already exists, proceeding to login...")
        else:
            resp.raise_for_status()
            print("✅ Registration successful")
    except Exception as e:
        print(f"⚠️ Registration issue: {e}")

    print("Logging in...")
    try:
        resp = requests.post(
            f"{BASE_URL}/auth/login", json={"username": username, "password": password}
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        print("✅ Login successful. Token obtained.\n")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    # 2. Ask a Question
    question = "What is the capital of France?"
    print(f"❓ Asking: '{question}'")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 3},
            headers=headers,
        )
        resp.raise_for_status()
        result = resp.json()

        print("\n🤖 RAG Response:")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)
        print(f"\nFaithfulness Score: {result.get('confidence_score', 'N/A')}")
        print(f"Latency: {result.get('latency_ms', 0):.2f} ms")

    except Exception as e:
        print(f"❌ Query failed: {e}")


if __name__ == "__main__":
    main()
