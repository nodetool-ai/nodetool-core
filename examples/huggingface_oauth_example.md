# Hugging Face OAuth Integration Example

This document demonstrates how to use the Hugging Face OAuth integration in NodeTool.

## Overview

The OAuth integration provides a secure way to connect Hugging Face accounts to NodeTool, allowing users to:
- Access gated models and datasets
- Use Hugging Face Spaces
- Make authenticated API requests to Hugging Face

## Architecture

The implementation follows the OAuth 2.0 PKCE (Proof Key for Code Exchange) flow:

1. Client requests authorization URL from `/oauth/hf/start`
2. Client opens system browser to Hugging Face authorization page
3. User authorizes the application
4. Hugging Face redirects to `/oauth/hf/callback` with authorization code
5. Backend exchanges code for access token and stores it encrypted
6. Client can now use the token to access Hugging Face APIs

## API Endpoints

### 1. Start OAuth Flow

**GET** `/oauth/hf/start`

Initiates the OAuth flow by generating PKCE challenge and returning the authorization URL.

**Request:**
```bash
curl -X GET http://127.0.0.1:8000/oauth/hf/start \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

**Response:**
```json
{
  "auth_url": "https://huggingface.co/oauth/authorize?client_id=nodetool-local&redirect_uri=http%3A%2F%2F127.0.0.1%3A8000%2Foauth%2Fhf%2Fcallback&response_type=code&scope=read+write+inference&state=xyz&code_challenge=abc&code_challenge_method=S256"
}
```

### 2. OAuth Callback

**GET** `/oauth/hf/callback`

Handles the OAuth callback from Hugging Face. This endpoint is called by the browser after user authorization.

**Parameters:**
- `code`: Authorization code from Hugging Face
- `state`: State parameter for CSRF protection
- `error`: (optional) Error from OAuth provider
- `error_description`: (optional) Error description

**Response:**
Returns an HTML page showing success or error message.

### 3. List Tokens

**GET** `/oauth/hf/tokens`

Lists all stored Hugging Face OAuth tokens for the current user.

**Request:**
```bash
curl -X GET http://127.0.0.1:8000/oauth/hf/tokens \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

**Response:**
```json
{
  "tokens": [
    {
      "id": "credential_id",
      "provider": "huggingface",
      "account_id": "user123",
      "username": "testuser",
      "token_type": "Bearer",
      "scope": "read write inference",
      "received_at": "2025-01-01T00:00:00Z",
      "expires_at": "2025-01-31T00:00:00Z",
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-01-01T00:00:00Z"
    }
  ]
}
```

### 4. Refresh Token

**POST** `/oauth/hf/refresh?account_id=ACCOUNT_ID`

Refreshes an access token using the stored refresh token.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/oauth/hf/refresh?account_id=user123" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

**Response:**
```json
{
  "success": true,
  "message": "Token refreshed successfully"
}
```

### 5. Get Account Info (Example)

**GET** `/oauth/hf/whoami?account_id=ACCOUNT_ID`

Demonstrates using the stored token to make authenticated requests to Hugging Face API.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/oauth/hf/whoami?account_id=user123" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

**Response:**
```json
{
  "id": "user123",
  "name": "Test User",
  "email": "test@example.com",
  "type": "user",
  "orgs": []
}
```

## Electron Integration Example

Here's how to integrate the OAuth flow in an Electron application:

```javascript
const { shell } = require('electron');
const axios = require('axios');

const API_BASE = 'http://127.0.0.1:8000';
const AUTH_TOKEN = 'your_nodetool_auth_token';

async function authenticateHuggingFace() {
  try {
    // 1. Start OAuth flow
    const startResponse = await axios.get(`${API_BASE}/oauth/hf/start`, {
      headers: { 'Authorization': `Bearer ${AUTH_TOKEN}` }
    });
    
    const authUrl = startResponse.data.auth_url;
    
    // 2. Open system browser
    await shell.openExternal(authUrl);
    
    // 3. Poll for completion (or use other notification mechanism)
    const accountId = await pollForCompletion();
    
    console.log('Authentication successful!', accountId);
    
    // 4. Test the connection
    const whoami = await axios.get(
      `${API_BASE}/oauth/hf/whoami?account_id=${accountId}`,
      { headers: { 'Authorization': `Bearer ${AUTH_TOKEN}` } }
    );
    
    console.log('Connected to Hugging Face:', whoami.data);
    
  } catch (error) {
    console.error('Authentication failed:', error);
  }
}

async function pollForCompletion() {
  // Poll the tokens endpoint until we see a new token
  const maxAttempts = 60; // 5 minutes
  
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await axios.get(`${API_BASE}/oauth/hf/tokens`, {
        headers: { 'Authorization': `Bearer ${AUTH_TOKEN}` }
      });
      
      if (response.data.tokens.length > 0) {
        // Found a token!
        return response.data.tokens[0].account_id;
      }
    } catch (error) {
      console.error('Error polling for tokens:', error);
    }
    
    // Wait 5 seconds before next attempt
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  
  throw new Error('Authentication timeout - user may not have completed authorization');
}

// Call the function
authenticateHuggingFace();
```

## Python Helper Functions

The `oauth_helper.py` module provides convenient functions for working with OAuth credentials:

```python
from nodetool.security.oauth_helper import (
    list_huggingface_accounts,
    get_huggingface_token,
    refresh_huggingface_token,
    get_huggingface_whoami,
)

# List all connected Hugging Face accounts
accounts = await list_huggingface_accounts(user_id)
# Returns: [{"account_id": "...", "username": "...", ...}, ...]

# Get access token for a specific account
token = await get_huggingface_token(user_id, account_id)
# Returns: "hf_xxxxxxxxxxxxx" or None

# Refresh an access token
success = await refresh_huggingface_token(user_id, account_id)
# Returns: True if refresh succeeded, False otherwise

# Get account information
info = await get_huggingface_whoami(user_id, account_id)
# Returns: {"id": "...", "name": "...", "email": "...", ...}
```

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "error_code",
  "error_description": "Human-readable error description"
}
```

Common error codes:
- `invalid_state`: State parameter is invalid or expired
- `token_exchange_failed`: Failed to exchange authorization code for tokens
- `refresh_failed`: Token refresh failed
- `network_error`: Network communication error
- `unauthorized`: Token is expired or invalid
- `internal_error`: Unexpected server error

## Security Considerations

1. **Token Storage**: All tokens are encrypted using the master key and user_id as salt
2. **PKCE Flow**: Uses PKCE to prevent authorization code interception
3. **State Validation**: Validates state parameter to prevent CSRF attacks
4. **Local-only**: Binds to 127.0.0.1 for local development security
5. **No Token Logging**: Tokens are never logged to prevent exposure
6. **Automatic Expiration**: State values expire after 5 minutes

## Multiple Accounts

The system supports multiple Hugging Face accounts per user. Each account is identified by its `account_id` (Hugging Face user ID).

To add multiple accounts, simply repeat the OAuth flow. Each new authorization will create a separate credential entry.

## Token Refresh

Access tokens may expire. When this happens:

1. API calls will return a 401 Unauthorized error
2. Call `/oauth/hf/refresh?account_id=XXX` to get a new access token
3. If refresh fails (no refresh token or refresh token expired), user must re-authenticate

## Testing

Run the OAuth tests:

```bash
pytest tests/api/test_oauth.py -v
```

All tests use mocked HTTP requests and don't require actual Hugging Face credentials.
