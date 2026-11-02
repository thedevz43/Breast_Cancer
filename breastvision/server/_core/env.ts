import crypto from "crypto";

const isProduction = process.env.NODE_ENV === "production";

// Ensure cookie secret is never an insecure empty string
const getSecureCookieSecret = (): string => {
  const secret = process.env.JWT_SECRET;
  if (secret && secret.trim().length >= 16) {
    return secret;
  }
  if (isProduction) {
    console.error(
      "[SECURITY ALERT] JWT_SECRET environment variable is missing or too short! Generating ephemeral 256-bit key for process lifetime.",
    );
  }
  // Generate a cryptographically strong 256-bit random key if not configured
  return crypto.randomBytes(32).toString("hex");
};

export const ENV = {
  appId: process.env.VITE_APP_ID ?? "",
  cookieSecret: getSecureCookieSecret(),
  databaseUrl: process.env.DATABASE_URL ?? "",
  mongodbUri: process.env.MONGODB_URI ?? process.env.MONGO_URL ?? "",
  oAuthServerUrl: process.env.OAUTH_SERVER_URL ?? "",
  ownerOpenId: process.env.OWNER_OPEN_ID ?? "",
  isProduction,
  forgeApiUrl: process.env.BUILT_IN_FORGE_API_URL ?? "",
  forgeApiKey: process.env.BUILT_IN_FORGE_API_KEY ?? "",
};
