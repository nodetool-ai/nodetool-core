# GitHub OAuth Implementation

This document describes the GitHub OAuth implementation with PKCE support for NodeTool.

## Overview

The implementation provides a complete OAuth2 flow for GitHub authentication with the following features:

- Authorization code flow with PKCE security
- Local-first design (binds to 127.0.0.1 only)
- Multi-account support
- Token persistence across app restarts
- Secure token storage using database encryption
- Helper functions for GitHub API access

## Architecture

### Components

1. **OAuthToken Model** (`src/nodetool/models/oauth_token.py`)
   - Database model for storing OAuth credentials
   - Fields: provider, access_token, refresh_token, token_type, scope, received_at, expires_at, account_id
   - Encrypted storage via database

2. **OAuth API** (`src/nodetool/api/oauth.py`)
   - FastAPI endpoints for OAuth flow
   - State management for CSRF protection
   - PKCE code verifier and challenge generation
   - Token exchange with GitHub

3. **Database Migration** (`src/nodetool/models/migrations/20251225000000_create_oauth_tokens.sql`)
   - Creates `nodetool_oauth_tokens` table
   - Indexes for efficient querying

## Setup

### 1. Configure GitHub OAuth App

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Fill in:
   - Application name: "NodeTool Local"
   - Homepage URL: "http://localhost:8000"
   - Authorization callback URL: "http://127.0.0.1:8000/api/oauth/github/callback"
4. Copy the Client ID and Client Secret

### 2. Configure NodeTool

Add to your `.env.development.local` or `.env.local`:

```bash
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
```

## API Endpoints

### 1. Start OAuth Flow

**GET** `/api/oauth/github/start`

Initiates the GitHub OAuth flow by generating an authorization URL with PKCE.

**Response:**
```json
{
  "auth_url": "https://github.com/login/oauth/authorize?client_id=...&redirect_uri=...&scope=...&state=..."
}
```

**Scopes requested:**
- `repo` - Full control of private repositories
- `workflow` - Update GitHub Action workflows
- `read:user` - Read user profile data
- `user:email` - Access user email addresses

### 2. OAuth Callback

**GET** `/api/oauth/github/callback`

Handles the OAuth callback from GitHub after user authorization.

**Query Parameters:**
- `code` - Authorization code from GitHub
- `state` - CSRF protection token

**Response:**
- HTML page with success/error message that auto-closes

### 3. List GitHub Tokens

**GET** `/api/oauth/github/tokens`

Lists all stored GitHub OAuth tokens for the current user.

**Response:**
```json
{
  "tokens": [
    {
      "id": "...",
      "provider": "github",
      "account_id": "12345",
      "token_type": "bearer",
      "scope": "repo,workflow,read:user,user:email",
      "has_refresh_token": false,
      "received_at": "2025-12-25T00:00:00",
      "expires_at": null,
      "is_expired": false,
      "created_at": "2025-12-25T00:00:00",
      "updated_at": "2025-12-25T00:00:00"
    }
  ]
}
```

### 4. Refresh Token

**POST** `/api/oauth/github/refresh?account_id={account_id}`

**Note:** GitHub OAuth tokens do not expire and cannot be refreshed. This endpoint returns an error for API consistency.

### 5. Revoke Token

**DELETE** `/api/oauth/github/tokens/{account_id}`

Removes a stored GitHub token from the database.

## Electron Integration

### Frontend Flow

```typescript
// 1. Call the start endpoint
const response = await fetch('http://127.0.0.1:8000/api/oauth/github/start', {
  headers: {
    'Authorization': `Bearer ${authToken}`
  }
});
const { auth_url } = await response.json();

// 2. Open the URL in the system browser
const { shell } = require('electron');
shell.openExternal(auth_url);

// 3. Poll for token completion
const checkInterval = setInterval(async () => {
  const tokensResponse = await fetch('http://127.0.0.1:8000/api/oauth/github/tokens', {
    headers: {
      'Authorization': `Bearer ${authToken}`
    }
  });
  const { tokens } = await tokensResponse.json();
  
  if (tokens.length > 0) {
    clearInterval(checkInterval);
    console.log('GitHub authentication successful!');
  }
}, 2000);
```

## Helper Functions

### List GitHub Accounts

```python
from nodetool.api.oauth import list_github_accounts

accounts = await list_github_accounts(user_id="user123")
```

### Get GitHub Token

```python
from nodetool.api.oauth import get_github_token

access_token = await get_github_token(user_id="user123", account_id="12345")
```

## Usage Example: Accessing GitHub API

```python
import httpx
from nodetool.models.oauth_token import OAuthToken

async def list_user_repositories(user_id: str, account_id: str):
    """List all repositories for the authenticated GitHub user."""
    token = await OAuthToken.find_by_account(user_id, "github", account_id)
    if not token:
        raise ValueError("GitHub account not connected")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/user/repos",
            headers={
                "Authorization": f"Bearer {token.access_token}",
                "Accept": "application/vnd.github.v3+json"
            },
            params={"per_page": 100, "sort": "updated"}
        )
        
        if response.status_code != 200:
            raise Exception(f"GitHub API error: {response.text}")
        
        return response.json()
```

## Security Considerations

- OAuth tokens are stored in the database table `nodetool_oauth_tokens`
- State parameter is generated with secure random values
- State expires after 10 minutes (default TTL)
- PKCE parameters are generated for forward compatibility
- Token revocation on GitHub requires manual action via GitHub settings

## Error Handling

Error codes: `invalid_state`, `token_exchange_failed`, `expired_code`, `network_error`, `configuration_error`

## Testing

```bash
pytest tests/api/test_oauth.py -v
```

## References

- [GitHub OAuth Documentation](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)
- [OAuth 2.0 RFC](https://tools.ietf.org/html/rfc6749)
- [GitHub REST API](https://docs.github.com/en/rest)
