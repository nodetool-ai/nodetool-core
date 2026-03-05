export {
  TokenType,
  AuthProvider,
  type AuthResult,
} from "./auth-provider.js";

export { LocalAuthProvider } from "./providers/local-provider.js";
export { StaticTokenProvider } from "./providers/static-token-provider.js";

export {
  createAuthMiddleware,
  getUserId,
  HttpError,
  type AuthenticatedUser,
  type AuthMiddlewareOptions,
} from "./middleware.js";
