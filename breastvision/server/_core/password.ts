import crypto from "crypto";

/**
 * Enterprise Secure Password Hashing Module
 * Standard: Argon2 / Scrypt (OWASP recommended memory-hard hashing)
 * Default: Node.js native crypto.scrypt with 32-byte salt, N=16384, r=8, p=1
 */

const KEY_LENGTH = 64;
const SALT_LENGTH = 32;

/**
 * Hash a plain text password using cryptographic scrypt/argon2 standard.
 * Format: scrypt$<salt_hex>$<derived_key_hex>
 */
export async function hashPassword(plainPassword: string): Promise<string> {
  if (!plainPassword || typeof plainPassword !== "string" || plainPassword.length < 8) {
    throw new Error("Password must be at least 8 characters long.");
  }

  const salt = crypto.randomBytes(SALT_LENGTH).toString("hex");

  return new Promise((resolve, reject) => {
    crypto.scrypt(plainPassword, salt, KEY_LENGTH, { N: 16384, r: 8, p: 1, maxmem: 32 * 1024 * 1024 }, (err, derivedKey) => {
      if (err) return reject(err);
      resolve(`scrypt$${salt}$${derivedKey.toString("hex")}`);
    });
  });
}

/**
 * Verify a plain text password against a stored hash using constant-time comparison.
 */
export async function verifyPassword(plainPassword: string, storedHash: string): Promise<boolean> {
  if (!plainPassword || !storedHash) return false;

  const parts = storedHash.split("$");
  if (parts.length !== 3 || parts[0] !== "scrypt") {
    // Unsupported or corrupted hash format
    return false;
  }

  const [, salt, originalKeyHex] = parts;
  const originalKeyBuffer = Buffer.from(originalKeyHex, "hex");

  return new Promise((resolve) => {
    crypto.scrypt(plainPassword, salt, originalKeyBuffer.length, { N: 16384, r: 8, p: 1, maxmem: 32 * 1024 * 1024 }, (err, derivedKey) => {
      if (err) return resolve(false);
      // Constant-time buffer comparison to prevent timing attacks
      const match = crypto.timingSafeEqual(originalKeyBuffer, derivedKey);
      resolve(match);
    });
  });
}

