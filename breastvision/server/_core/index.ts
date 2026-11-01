import "dotenv/config";
import express from "express";
import { createServer } from "http";
import net from "net";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { registerStorageProxy } from "./storageProxy";
import { appRouter } from "../routers";
import { createContext } from "./context";

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(port, () => {
      server.close(() => resolve(true));
    });
    server.on("error", () => resolve(false));
  });
}

async function findAvailablePort(startPort: number = 3000): Promise<number> {
  for (let port = startPort; port < startPort + 20; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

async function startServer() {
  const app = express();
  const server = createServer(app);

  // 1. Security Headers (OWASP Recommended)
  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    res.setHeader("X-XSS-Protection", "1; mode=block");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    res.setHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
    res.setHeader(
      "Content-Security-Policy",
      "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; connect-src 'self' https:;",
    );
    if (process.env.NODE_ENV === "production") {
      res.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload");
    }
    next();
  });

  // 2. Strict CORS Configuration - Prevent Cross-Origin Credential Theft
  const isAllowedOrigin = (origin: string): boolean => {
    if (!origin) return false;
    try {
      const parsed = new URL(origin);
      const host = parsed.hostname.toLowerCase();
      // Allow localhost / loopback for dev
      if (host === "localhost" || host === "127.0.0.1" || host === "::1") return true;
      // Allow production / custom domains
      if (host.endsWith("runs-on.dev") || host.endsWith("onrender.com") || host.endsWith("vercel.app")) {
        return true;
      }
      const configuredOrigins = (process.env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim().toLowerCase());
      if (configuredOrigins.includes(origin.toLowerCase())) return true;
    } catch {
      return false;
    }
    return false;
  };

  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && isAllowedOrigin(origin)) {
      res.header("Access-Control-Allow-Origin", origin);
      res.header("Access-Control-Allow-Credentials", "true");
    }
    res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, X-Requested-With, Content-Type, Accept, Authorization",
    );

    // Handle preflight requests
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });

  // 3. In-memory Rate Limiting for API and Auth Endpoints
  const rateLimitStore = new Map<string, { count: number; resetTime: number }>();
  const createRateLimiter = (maxRequests: number, windowMs: number) => {
    return (req: express.Request, res: express.Response, next: express.NextFunction) => {
      const ip = req.ip || req.socket.remoteAddress || "unknown";
      const key = `${req.baseUrl || ""}:${ip}`;
      const now = Date.now();
      const record = rateLimitStore.get(key);

      if (!record || now > record.resetTime) {
        rateLimitStore.set(key, { count: 1, resetTime: now + windowMs });
        return next();
      }

      record.count += 1;
      if (record.count > maxRequests) {
        const retryAfterSec = Math.ceil((record.resetTime - now) / 1000);
        res.setHeader("Retry-After", String(retryAfterSec));
        res.status(429).json({
          error: "Too Many Requests",
          message: `Rate limit exceeded. Please wait ${retryAfterSec} seconds before retrying.`,
        });
        return;
      }
      next();
    };
  };

  const generalApiLimiter = createRateLimiter(120, 60 * 1000); // 120 req / min
  const authRateLimiter = createRateLimiter(15, 60 * 1000); // 15 auth req / min

  app.use(express.json({ limit: "15mb" }));
  app.use(express.urlencoded({ limit: "15mb", extended: true }));

  // Apply rate limiters
  app.use("/api/auth", authRateLimiter);
  app.use("/api/oauth", authRateLimiter);
  app.use("/api", generalApiLimiter);

  registerStorageProxy(app);
  registerOAuthRoutes(app);

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, timestamp: Date.now() });
  });

  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    }),
  );

  // 4. Global Error Handler - Hide Stack Traces & Error Details from Users
  app.use((err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const errorObj = err instanceof Error ? err : new Error(String(err));
    console.error("[ServerError:Caught]", errorObj.message);
    if (res.headersSent) return;
    res.status(500).json({
      error: "Internal Server Error",
      message: "An unexpected error occurred. Please contact the administrator or retry.",
    });
  });

  const preferredPort = parseInt(process.env.PORT || "3000");
  const port = await findAvailablePort(preferredPort);

  if (port !== preferredPort) {
    console.log(`Port ${preferredPort} is busy, using port ${port} instead`);
  }

  server.listen(port, () => {
    console.log(`[api] server listening on port ${port}`);
  });
}

startServer().catch(console.error);
