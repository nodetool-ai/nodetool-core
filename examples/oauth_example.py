#!/usr/bin/env python3
"""
Example script demonstrating the provider-agnostic OAuth integration.

This script shows how to:
1. Start OAuth flow for any provider (Google, GitHub, HF, OpenRouter)
2. Wait for user authentication
3. Retrieve token metadata
4. Use tokens to access provider APIs
5. Refresh tokens when they expire
6. Support multiple accounts per provider

Prerequisites:
- Set provider client IDs in your environment (e.g., GOOGLE_CLIENT_ID)
- NodeTool API server running on http://127.0.0.1:8000
"""

import asyncio
import sys
import webbrowser
from typing import Optional

import httpx


API_BASE = "http://127.0.0.1:8000"


async def list_providers() -> list[str]:
    """List all available OAuth providers."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE}/api/oauth/providers")
        response.raise_for_status()
        return response.json()["providers"]


async def start_oauth_flow(provider: str) -> dict:
    """
    Start the OAuth flow for a provider and wait for user to complete authentication.

    Args:
        provider: Provider name (google, github, hf, openrouter)

    Returns:
        dict: OAuth token metadata
    """
    async with httpx.AsyncClient() as client:
        # Step 1: Start OAuth flow
        print(f"\nStarting OAuth flow for {provider}...")
        response = await client.get(f"{API_BASE}/api/oauth/{provider}/start")
        response.raise_for_status()
        data = response.json()

        auth_url = data["auth_url"]
        state = data["state"]

        print(f"Opening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")

        # Open browser for user authentication
        webbrowser.open(auth_url)

        # Step 2: Poll for token metadata (wait for callback to complete)
        print("Waiting for authentication (60 second timeout)...")
        token_meta: Optional[dict] = None

        for attempt in range(60):
            await asyncio.sleep(1)

            try:
                response = await client.get(f"{API_BASE}/api/oauth/{provider}/tokens")
                if response.status_code == 200:
                    token_meta = response.json()
                    break
                elif response.status_code == 404:
                    # Not ready yet, keep polling
                    if attempt % 5 == 0 and attempt > 0:
                        print(f"Still waiting... ({attempt + 1}s elapsed)")
                    continue
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    break
            except httpx.HTTPError as e:
                print(f"HTTP error: {e}")
                break

        if not token_meta:
            raise Exception(f"OAuth timeout - user did not complete authentication for {provider}")

        print(f"✓ Authentication successful for {provider}!")
        return token_meta


async def refresh_tokens(provider: str, account_id: Optional[str] = None) -> dict:
    """
    Refresh access tokens using the refresh token.

    Args:
        provider: Provider name
        account_id: Optional account identifier

    Returns:
        dict: New OAuth token metadata
    """
    async with httpx.AsyncClient() as client:
        params = {"account_id": account_id} if account_id else {}
        print(f"Refreshing tokens for {provider}...")
        response = await client.post(f"{API_BASE}/api/oauth/{provider}/refresh", params=params)
        response.raise_for_status()
        token_meta = response.json()
        print(f"✓ Tokens refreshed for {provider}!")
        return token_meta


async def revoke_tokens(provider: str, account_id: Optional[str] = None):
    """Revoke (clear) stored OAuth tokens."""
    async with httpx.AsyncClient() as client:
        params = {"account_id": account_id} if account_id else {}
        print(f"Revoking tokens for {provider}...")
        response = await client.delete(f"{API_BASE}/api/oauth/{provider}/tokens", params=params)
        if response.status_code == 200:
            print(f"✓ Tokens revoked for {provider}!")
        else:
            print(f"Failed to revoke tokens: {response.status_code}")


async def demo_google_api(token_meta: dict):
    """Demonstrate using Google Sheets API (requires internal token access)."""
    print("\n" + "=" * 50)
    print("Google API Demo")
    print("=" * 50)

    # Note: For actual API calls, you'd need to get the access token
    # from the server-side code using get_access_token()
    # This is just metadata display for the example

    print(f"\nAccount: {token_meta.get('account_id', 'unknown')}")
    print(f"Scopes: {token_meta['scope']}")
    print(f"Expires at: {token_meta.get('expires_at', 'N/A')}")
    print(f"Needs refresh: {token_meta['needs_refresh']}")


async def demo_github_api(token_meta: dict):
    """Demonstrate using GitHub API."""
    print("\n" + "=" * 50)
    print("GitHub API Demo")
    print("=" * 50)

    print(f"\nAccount: {token_meta.get('account_id', 'unknown')}")
    print(f"Scopes: {token_meta['scope']}")
    print(f"Expires at: {token_meta.get('expires_at', 'N/A')}")


async def demo_provider(provider: str):
    """Demo OAuth flow for a specific provider."""
    print("\n" + "=" * 70)
    print(f"OAuth Demo: {provider.upper()}")
    print("=" * 70)

    try:
        # Authenticate
        token_meta = await start_oauth_flow(provider)

        # Display token info
        print("\nToken Metadata:")
        print(f"  Provider: {token_meta['provider']}")
        print(f"  Account: {token_meta.get('account_id', 'N/A')}")
        print(f"  Token Type: {token_meta['token_type']}")
        print(f"  Scopes: {token_meta['scope']}")
        print(f"  Received At: {token_meta['received_at']}")
        print(f"  Expires At: {token_meta.get('expires_at', 'N/A')}")
        print(f"  Is Expired: {token_meta['is_expired']}")
        print(f"  Needs Refresh: {token_meta['needs_refresh']}")

        # Provider-specific demos
        if provider == "google":
            await demo_google_api(token_meta)
        elif provider == "github":
            await demo_github_api(token_meta)

        # Demonstrate token refresh
        if token_meta.get('account_id'):
            print("\n" + "=" * 50)
            print("Demonstrating token refresh...")
            await refresh_tokens(provider, token_meta['account_id'])

        return True

    except Exception as e:
        print(f"\nError with {provider}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function demonstrating OAuth flow for multiple providers."""
    print("=" * 70)
    print("Provider-Agnostic OAuth Integration Example")
    print("=" * 70)

    # List available providers
    providers = await list_providers()
    print(f"\nAvailable providers: {', '.join(providers)}")

    # Check command line arguments
    if len(sys.argv) > 1:
        # Demo specific provider
        provider = sys.argv[1].lower()
        if provider not in providers:
            print(f"\nError: Unknown provider '{provider}'")
            print(f"Available: {', '.join(providers)}")
            return
        await demo_provider(provider)
    else:
        # Interactive mode
        print("\nWhich provider would you like to test?")
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
        print(f"  {len(providers) + 1}. Test all providers")
        print("  0. Exit")

        try:
            choice = int(input("\nEnter choice: "))
            if choice == 0:
                print("Exiting...")
                return
            elif choice == len(providers) + 1:
                # Test all providers
                for provider in providers:
                    success = await demo_provider(provider)
                    if not success:
                        print(f"\nSkipping remaining providers due to error with {provider}")
                        break
            elif 1 <= choice <= len(providers):
                provider = providers[choice - 1]
                await demo_provider(provider)
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Usage:")
    print("  python examples/oauth_example.py           # Interactive mode")
    print("  python examples/oauth_example.py google    # Test specific provider")
    print("=" * 70)

    asyncio.run(main())
