#!/usr/bin/env python3
"""
Example script demonstrating how to use the Google OAuth integration.

This script shows how to:
1. Start the OAuth flow
2. Wait for user authentication
3. Retrieve tokens
4. Use tokens to access Google APIs
5. Refresh tokens when they expire

Prerequisites:
- Set GOOGLE_CLIENT_ID in your environment
- NodeTool API server running on http://127.0.0.1:8000
"""

import asyncio
import webbrowser
from typing import Optional

import httpx


API_BASE = "http://127.0.0.1:8000"


async def start_oauth_flow() -> dict:
    """
    Start the OAuth flow and wait for user to complete authentication.

    Returns:
        dict: OAuth tokens including access_token, refresh_token, etc.
    """
    async with httpx.AsyncClient() as client:
        # Step 1: Start OAuth flow
        print("Starting OAuth flow...")
        response = await client.get(f"{API_BASE}/api/oauth/start")
        response.raise_for_status()
        data = response.json()

        auth_url = data["auth_url"]
        state = data["state"]

        print(f"Opening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")

        # Open browser for user authentication
        webbrowser.open(auth_url)

        # Step 2: Poll for tokens (wait for callback to complete)
        print("Waiting for authentication (60 second timeout)...")
        tokens: Optional[dict] = None

        for attempt in range(60):
            await asyncio.sleep(1)

            try:
                response = await client.get(f"{API_BASE}/api/oauth/tokens")
                if response.status_code == 200:
                    tokens = response.json()
                    break
                elif response.status_code == 404:
                    # Not ready yet, keep polling
                    if attempt % 5 == 0:
                        print(f"Still waiting... ({attempt + 1}s elapsed)")
                    continue
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    break
            except httpx.HTTPError as e:
                print(f"HTTP error: {e}")
                break

        if not tokens:
            raise Exception("OAuth timeout - user did not complete authentication")

        print("✓ Authentication successful!")
        return tokens


async def refresh_tokens() -> dict:
    """
    Refresh access tokens using the refresh token.

    Returns:
        dict: New OAuth tokens
    """
    async with httpx.AsyncClient() as client:
        print("Refreshing tokens...")
        response = await client.post(f"{API_BASE}/api/oauth/refresh")
        response.raise_for_status()
        tokens = response.json()
        print("✓ Tokens refreshed!")
        return tokens


async def list_spreadsheets(access_token: str) -> list:
    """
    List Google Sheets spreadsheets using the Drive API.

    Args:
        access_token: OAuth access token

    Returns:
        list: List of spreadsheet files
    """
    async with httpx.AsyncClient() as client:
        print("Fetching list of spreadsheets...")
        response = await client.get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": "mimeType='application/vnd.google-apps.spreadsheet'",
                "pageSize": 10,
                "fields": "files(id, name, createdTime)",
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("files", [])


async def get_spreadsheet_data(spreadsheet_id: str, access_token: str) -> dict:
    """
    Get data from a specific Google Sheets spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet
        access_token: OAuth access token

    Returns:
        dict: Spreadsheet data
    """
    async with httpx.AsyncClient() as client:
        print(f"Fetching spreadsheet {spreadsheet_id}...")
        response = await client.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        return response.json()


async def revoke_tokens():
    """Revoke (clear) stored OAuth tokens."""
    async with httpx.AsyncClient() as client:
        print("Revoking tokens...")
        response = await client.delete(f"{API_BASE}/api/oauth/tokens")
        if response.status_code == 200:
            print("✓ Tokens revoked!")
        else:
            print(f"Failed to revoke tokens: {response.status_code}")


async def main():
    """Main function demonstrating OAuth flow and API usage."""
    try:
        # Authenticate with Google
        tokens = await start_oauth_flow()

        print("\nToken Info:")
        print(f"  Access Token: {tokens['access_token'][:20]}...")
        print(f"  Token Type: {tokens['token_type']}")
        print(f"  Expires In: {tokens['expires_in']} seconds")
        print(f"  Scope: {tokens['scope']}")

        # Use the access token to list spreadsheets
        print("\n" + "=" * 50)
        spreadsheets = await list_spreadsheets(tokens["access_token"])

        if spreadsheets:
            print(f"\nFound {len(spreadsheets)} spreadsheet(s):")
            for sheet in spreadsheets:
                print(f"  - {sheet['name']} (ID: {sheet['id']})")

            # Optionally, fetch data from the first spreadsheet
            # first_sheet = spreadsheets[0]
            # data = await get_spreadsheet_data(first_sheet['id'], tokens['access_token'])
            # print(f"\nSpreadsheet '{first_sheet['name']}' has {len(data.get('sheets', []))} sheet(s)")
        else:
            print("\nNo spreadsheets found in your Google Drive")

        # Demonstrate token refresh
        print("\n" + "=" * 50)
        print("\nDemonstrating token refresh...")
        new_tokens = await refresh_tokens()
        print(f"New Access Token: {new_tokens['access_token'][:20]}...")

        # Clean up (optional)
        # await revoke_tokens()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("Google OAuth Integration Example")
    print("=" * 50)
    print()

    asyncio.run(main())
