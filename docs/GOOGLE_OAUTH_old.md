# Google OAuth Integration Guide

This guide explains how to use the Google OAuth 2.0 integration in NodeTool for accessing Google APIs like Sheets and Drive.

## Overview

NodeTool implements Google OAuth 2.0 with PKCE (Proof Key for Code Exchange) for secure authentication. The OAuth flow is handled entirely by the FastAPI backend, keeping secrets and tokens secure.

## Architecture

```
Electron/Client → FastAPI Backend → Google OAuth → Google APIs
```

1. Client requests OAuth start
2. Backend generates PKCE challenge and returns authorization URL
3. User authenticates with Google in their browser
4. Google redirects back to backend callback
5. Backend exchanges code for tokens
6. Client retrieves tokens for API calls

## Setup

### 1. Get Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the APIs you need (Sheets API, Drive API, etc.)
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
5. Choose "Web application" as the application type
6. Add authorized redirect URI: `http://127.0.0.1:8000/api/oauth/callback`
   - Replace `8000` with your actual API port if different
7. Copy the Client ID (you don't need the Client Secret for PKCE flow)

### 2. Configure Environment

Add to your `.env.development.local` or `.env.production.local`:

```bash
GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
# Note: GOOGLE_CLIENT_SECRET is optional for PKCE flow but registered for completeness
```

Or set via the settings UI in development mode.

## API Endpoints

### `GET /api/oauth/start`

Initiates the OAuth flow.

**Response:**
```json
{
  "auth_url": "https://accounts.google.com/o/oauth2/v2/auth?...",
  "state": "random-state-string"
}
```

The client should open `auth_url` in the system browser.

### `GET /api/oauth/callback`

Handles the OAuth callback from Google. This is called automatically by Google after user authorization.

**Query Parameters:**
- `code`: Authorization code from Google
- `state`: State parameter for CSRF protection

**Response:** HTML page indicating success or failure

### `GET /api/oauth/tokens`

Retrieves the stored OAuth tokens.

**Response:**
```json
{
  "access_token": "ya29.a0...",
  "refresh_token": "1//0e...",
  "expires_in": 3600,
  "token_type": "Bearer",
  "scope": "https://www.googleapis.com/auth/spreadsheets ...",
  "received_at": 1234567890
}
```

Returns 404 if no tokens are available.

### `POST /api/oauth/refresh`

Refreshes the access token using the refresh token.

**Response:** Same as `/api/oauth/tokens`

### `DELETE /api/oauth/tokens`

Revokes the stored tokens (clears them from memory).

**Response:**
```json
{
  "message": "Tokens revoked successfully"
}
```

## Usage Example (Electron)

```javascript
const { shell } = require("electron");
const fetch = require("node-fetch");

const API_BASE = "http://127.0.0.1:8000";

async function authenticateWithGoogle() {
  try {
    // Step 1: Start OAuth flow
    const startResponse = await fetch(`${API_BASE}/api/oauth/start`);
    const { auth_url, state } = await startResponse.json();

    // Step 2: Open browser for user authentication
    await shell.openExternal(auth_url);

    // Step 3: Poll for tokens (callback happens automatically)
    let tokens = null;
    let attempts = 0;
    const maxAttempts = 60; // 60 seconds timeout

    while (!tokens && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const tokenResponse = await fetch(`${API_BASE}/api/oauth/tokens`);
      if (tokenResponse.ok) {
        tokens = await tokenResponse.json();
      }
      attempts++;
    }

    if (!tokens) {
      throw new Error("OAuth timeout - user did not complete authentication");
    }

    console.log("Authentication successful!");
    return tokens;

  } catch (error) {
    console.error("OAuth error:", error);
    throw error;
  }
}

// Example: Use tokens to access Google Sheets
async function listSpreadsheets(accessToken) {
  const response = await fetch(
    "https://www.googleapis.com/drive/v3/files?q=mimeType='application/vnd.google-apps.spreadsheet'",
    {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    }
  );

  return await response.json();
}
```

## Usage Example (Python)

```python
import httpx
import webbrowser
import time

API_BASE = "http://127.0.0.1:8000"

async def authenticate_with_google():
    async with httpx.AsyncClient() as client:
        # Start OAuth flow
        response = await client.get(f"{API_BASE}/api/oauth/start")
        data = response.json()

        # Open browser
        webbrowser.open(data["auth_url"])

        # Poll for tokens
        tokens = None
        for _ in range(60):  # 60 second timeout
            time.sleep(1)
            response = await client.get(f"{API_BASE}/api/oauth/tokens")
            if response.status_code == 200:
                tokens = response.json()
                break

        if not tokens:
            raise Exception("OAuth timeout")

        return tokens

# Example: Use tokens
async def get_spreadsheet_data(spreadsheet_id: str, access_token: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        return response.json()
```

## Token Management

### Access Token Expiration

Access tokens typically expire after 1 hour. Use the `/api/oauth/refresh` endpoint to get a new access token:

```javascript
async function refreshAccessToken() {
  const response = await fetch(`${API_BASE}/api/oauth/refresh`, {
    method: "POST",
  });

  if (!response.ok) {
    // Refresh failed - need to re-authenticate
    return await authenticateWithGoogle();
  }

  return await response.json();
}
```

### Token Storage

Currently, tokens are stored in memory on the backend. This means:
- ✓ Tokens are secure (not exposed to frontend)
- ✓ No risk of token leakage
- ✗ Tokens are lost when server restarts
- ✗ No multi-user support (single token store)

For production use, consider implementing:
- Database storage for tokens (encrypted)
- User-specific token storage
- OS keychain integration

## Security Considerations

1. **Localhost Only**: OAuth endpoints are only accessible from `127.0.0.1`
2. **PKCE Flow**: Uses code challenge/verifier to prevent authorization code interception
3. **State Validation**: Prevents CSRF attacks
4. **No Client Secret**: PKCE flow doesn't require client secret for public clients
5. **Token Storage**: Tokens stored server-side, never exposed to client
6. **Scopes**: Default scopes are minimal (Sheets + Drive readonly)

## Troubleshooting

### "OAuth not configured" error

Make sure `GOOGLE_CLIENT_ID` is set in your environment.

### Redirect URI mismatch

Ensure the redirect URI in Google Cloud Console exactly matches:
```
http://127.0.0.1:<port>/api/oauth/callback
```

### No refresh token received

Add `prompt=consent` to force consent screen, which ensures a refresh token is issued. This is already included in the implementation.

### Token expired

Use the `/api/oauth/refresh` endpoint to get a new access token without re-authentication.

## Scopes

Default scopes included:
- `https://www.googleapis.com/auth/spreadsheets` - Full access to Google Sheets
- `https://www.googleapis.com/auth/drive.readonly` - Read-only access to Google Drive

To modify scopes, edit the `DEFAULT_SCOPES` list in `src/nodetool/api/oauth.py`.

## Future Enhancements

Potential improvements for production:
1. Persistent token storage (database or keychain)
2. Multi-user support (user-specific token stores)
3. Token encryption at rest
4. Automatic token refresh before expiration
5. WebSocket notifications for OAuth completion
6. Support for additional OAuth providers (Microsoft, Dropbox, etc.)
7. Configurable scopes per authentication request
