# OAuth Integration Guide

This guide explains how to use the provider-agnostic OAuth 2.0 integration in NodeTool for accessing external APIs (Google Sheets/Drive, GitHub, Hugging Face, OpenRouter, etc.).

## Overview

NodeTool implements a **generic OAuth 2.0 engine** that works with multiple providers through a unified interface. The OAuth flow is handled entirely by the FastAPI backend, keeping secrets and tokens secure.

### Supported Providers

- **Google** - Sheets API, Drive API
- **GitHub** - Repository access, user information
- **Hugging Face** - Model access, repository management
- **OpenRouter** - AI model API access

Adding new providers requires only a configuration entry—no new endpoints or duplicated logic.

## Architecture

```
Client/Electron → FastAPI Backend → Provider Registry → OAuth Provider
                                        ↓
                                   Token Storage
                                   (multi-account)
```

**Flow:**
1. Client requests `/oauth/{provider}/start`
2. Backend generates PKCE challenge and returns authorization URL
3. User authenticates with provider in their browser
4. Provider redirects back to `/oauth/{provider}/callback`
5. Backend exchanges code for tokens and stores them
6. Client retrieves token metadata via `/oauth/{provider}/tokens`

## Setup

### 1. Get OAuth Credentials

#### Google

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select project
3. Enable APIs (Sheets API, Drive API, etc.)
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
5. Choose "Web application"
6. Add authorized redirect URI: `http://127.0.0.1:8000/api/oauth/google/callback`
7. Copy Client ID

#### GitHub

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Set Authorization callback URL: `http://127.0.0.1:8000/api/oauth/github/callback`
4. Copy Client ID and Client Secret

#### Hugging Face

1. Go to [Hugging Face Settings](https://huggingface.co/settings/applications)
2. Create new OAuth application
3. Set Redirect URI: `http://127.0.0.1:8000/api/oauth/hf/callback`
4. Copy Client ID and Client Secret

#### OpenRouter

1. Visit [OpenRouter](https://openrouter.ai/) OAuth settings
2. Register application with callback: `http://127.0.0.1:8000/api/oauth/openrouter/callback`
3. Copy Client ID and Client Secret

### 2. Configure Environment

Add to your `.env.development.local`:

```bash
# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Hugging Face OAuth
HF_OAUTH_CLIENT_ID=your-hf-client-id
HF_OAUTH_CLIENT_SECRET=your-hf-client-secret

# OpenRouter OAuth
OPENROUTER_CLIENT_ID=your-openrouter-client-id
OPENROUTER_CLIENT_SECRET=your-openrouter-client-secret
```

Or set via the settings UI in development mode.

## API Endpoints

All providers use the same endpoints. Just replace `{provider}` with `google`, `github`, `hf`, or `openrouter`.

### `GET /api/oauth/providers`

List all available OAuth providers.

**Response:**
```json
{
  "providers": ["google", "github", "hf", "openrouter"]
}
```

### `GET /api/oauth/{provider}/start`

Initiates the OAuth flow for a specific provider.

**Example:**
```bash
GET /api/oauth/google/start
GET /api/oauth/github/start
```

**Response:**
```json
{
  "auth_url": "https://accounts.google.com/o/oauth2/v2/auth?...",
  "state": "random-state-string",
  "provider": "google"
}
```

The client should open `auth_url` in the system browser.

### `GET /api/oauth/{provider}/callback`

Handles the OAuth callback from the provider. Called automatically by the provider after user authorization.

**Query Parameters:**
- `code`: Authorization code from provider
- `state`: State parameter for CSRF protection

**Response:** HTML page indicating success or failure

### `GET /api/oauth/{provider}/tokens`

Retrieves token metadata (not the actual tokens for security).

**Query Parameters:**
- `account_id` (optional): Specific account identifier for multi-account support

**Response:**
```json
{
  "provider": "google",
  "account_id": "user@example.com",
  "scope": "https://www.googleapis.com/auth/spreadsheets ...",
  "token_type": "Bearer",
  "received_at": 1234567890,
  "expires_at": 1234571490,
  "is_expired": false,
  "needs_refresh": false
}
```

Returns 404 if no tokens are available. Actual tokens are never returned for security.

### `POST /api/oauth/{provider}/refresh`

Refreshes the access token using the refresh token.

**Query Parameters:**
- `account_id` (optional): Specific account identifier

**Response:** Same as `/tokens` endpoint

### `DELETE /api/oauth/{provider}/tokens`

Revokes the stored tokens (clears them from memory).

**Query Parameters:**
- `account_id` (optional): Specific account identifier

**Response:**
```json
{
  "message": "Tokens revoked successfully for google",
  "provider": "google"
}
```

## Usage Examples

### Electron Integration

```javascript
const { shell } = require("electron");
const fetch = require("node-fetch");

const API_BASE = "http://127.0.0.1:8000";

async function authenticateWithProvider(provider) {
  try {
    // Step 1: Start OAuth flow
    const startResponse = await fetch(`${API_BASE}/api/oauth/${provider}/start`);
    const { auth_url, state } = await startResponse.json();

    // Step 2: Open browser for user authentication
    await shell.openExternal(auth_url);

    // Step 3: Poll for tokens (callback happens automatically)
    let tokenMeta = null;
    let attempts = 0;
    const maxAttempts = 60; // 60 seconds timeout

    while (!tokenMeta && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const tokenResponse = await fetch(`${API_BASE}/api/oauth/${provider}/tokens`);
      if (tokenResponse.ok) {
        tokenMeta = await tokenResponse.json();
      }
      attempts++;
    }

    if (!tokenMeta) {
      throw new Error("OAuth timeout - user did not complete authentication");
    }

    console.log(`Authentication successful for ${provider}!`);
    return tokenMeta;

  } catch (error) {
    console.error("OAuth error:", error);
    throw error;
  }
}

// Usage for different providers
await authenticateWithProvider("google");
await authenticateWithProvider("github");
await authenticateWithProvider("hf");
```

### Python Integration

```python
import httpx
import webbrowser
import time

API_BASE = "http://127.0.0.1:8000"

async def authenticate_with_provider(provider: str):
    async with httpx.AsyncClient() as client:
        # Start OAuth flow
        response = await client.get(f"{API_BASE}/api/oauth/{provider}/start")
        data = response.json()

        # Open browser
        webbrowser.open(data["auth_url"])

        # Poll for tokens
        token_meta = None
        for _ in range(60):  # 60 second timeout
            time.sleep(1)
            response = await client.get(f"{API_BASE}/api/oauth/{provider}/tokens")
            if response.status_code == 200:
                token_meta = response.json()
                break

        if not token_meta:
            raise Exception("OAuth timeout")

        return token_meta

# Usage
google_meta = await authenticate_with_provider("google")
github_meta = await authenticate_with_provider("github")
```

### Multi-Account Support

```python
# Authenticate multiple Google accounts
await authenticate_with_provider("google")  # account 1
# Repeat flow in browser with different account
await authenticate_with_provider("google")  # account 2

# Get specific account token
response = await client.get(
    f"{API_BASE}/api/oauth/google/tokens",
    params={"account_id": "user1@example.com"}
)
```

## Token Management

### Access Token Expiration

Access tokens typically expire after 1 hour. The API returns `needs_refresh: true` when token is close to expiry.

```javascript
async function refreshIfNeeded(provider, accountId = null) {
  const params = accountId ? `?account_id=${accountId}` : "";

  const response = await fetch(
    `${API_BASE}/api/oauth/${provider}/tokens${params}`
  );
  const tokenMeta = await response.json();

  if (tokenMeta.needs_refresh) {
    const refreshResponse = await fetch(
      `${API_BASE}/api/oauth/${provider}/refresh${params}`,
      { method: "POST" }
    );
    return await refreshResponse.json();
  }

  return tokenMeta;
}
```

### Token Storage

**Current Implementation:**
- ✓ Tokens stored in-memory on backend (secure)
- ✓ Never exposed to frontend
- ✓ Support for multiple accounts per provider
- ✗ Lost when server restarts
- ✗ Not encrypted at rest

**For Production:**
- Implement database storage with encryption
- Use OS keychain for local deployments
- Add user-specific token isolation

## Security Considerations

1. **PKCE Flow**: Uses code challenge/verifier (when supported by provider)
2. **State Validation**: Prevents CSRF attacks
3. **Localhost Only**: OAuth callbacks only work on 127.0.0.1
4. **Secure Random**: Uses `secrets` module for cryptographic operations
5. **Token Isolation**: Tokens stored server-side, never exposed to client
6. **URL Encoding**: Proper encoding prevents injection attacks
7. **Error Handling**: Unified error responses across providers

## Provider-Specific Details

### Google

- **Scopes**: `spreadsheets`, `drive.readonly`
- **Supports PKCE**: Yes
- **Supports Refresh**: Yes
- **Account ID**: Email address
- **Identity Endpoint**: `https://www.googleapis.com/oauth2/v2/userinfo`

### GitHub

- **Scopes**: `repo`, `user:email`
- **Supports PKCE**: Yes
- **Supports Refresh**: Yes
- **Account ID**: GitHub username
- **Identity Endpoint**: `https://api.github.com/user`

### Hugging Face

- **Scopes**: `read-repos`, `write-repos`
- **Supports PKCE**: Yes
- **Supports Refresh**: Yes
- **Account ID**: HF username
- **Identity Endpoint**: `https://huggingface.co/api/whoami`

### OpenRouter

- **Scopes**: `read`, `write`
- **Supports PKCE**: No (uses client secret)
- **Supports Refresh**: Yes
- **Account ID**: User ID or key
- **Identity Endpoint**: `https://openrouter.ai/api/v1/auth/key`

## Adding New Providers

To add a new OAuth provider:

1. **Add provider spec** to `src/nodetool/api/oauth_providers/spec.py`:

```python
PROVIDERS["newprovider"] = OAuthProviderSpec(
    name="newprovider",
    auth_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token",
    scopes=["read", "write"],
    client_id_env="NEWPROVIDER_CLIENT_ID",
    client_secret_env="NEWPROVIDER_CLIENT_SECRET",
    token_normalizer=normalize_newprovider_token,  # Optional
    identity_endpoint="https://provider.com/api/user",  # Optional
    supports_pkce=True,
    supports_refresh=True,
)
```

2. **Register secrets** in `src/nodetool/config/settings.py`:

```python
register_secret(
    package_name="nodetool",
    env_var="NEWPROVIDER_CLIENT_ID",
    group="NewProvider",
    description="NewProvider OAuth 2.0 Client ID",
)
```

3. **Update .env.example**:

```bash
NEWPROVIDER_CLIENT_ID=your_client_id
NEWPROVIDER_CLIENT_SECRET=your_client_secret
```

**That's it!** No new routes or endpoints needed. The provider automatically works with:
- `GET /api/oauth/newprovider/start`
- `GET /api/oauth/newprovider/callback`
- `GET /api/oauth/newprovider/tokens`
- `POST /api/oauth/newprovider/refresh`
- `DELETE /api/oauth/newprovider/tokens`

## Troubleshooting

### "OAuth not configured" error

Make sure the provider's client ID is set: `{PROVIDER}_CLIENT_ID`

### Redirect URI mismatch

Ensure redirect URI in provider settings exactly matches:
```
http://127.0.0.1:{port}/api/oauth/{provider}/callback
```

### No refresh token received

Some providers require specific parameters. Check provider's extra_auth_params in spec.

### Token expired

Use the `/oauth/{provider}/refresh` endpoint. If refresh fails, re-authenticate.

### Unknown provider

Check `/api/oauth/providers` for list of available providers.

## Internal Token Access

For server-side code that needs actual tokens:

```python
from nodetool.api.oauth import get_access_token

# Get token for internal use (not exposed via HTTP)
token = get_access_token("google", account_id="user@example.com")
if token:
    # Use token to call Google APIs
    pass
```

## Future Enhancements

Planned improvements:
1. Persistent database storage with encryption
2. User-specific token isolation (multi-user support)
3. Automatic token refresh before expiration
4. WebSocket notifications for OAuth completion
5. Token revocation with provider
6. Audit logging for token access
7. Rate limiting per provider
