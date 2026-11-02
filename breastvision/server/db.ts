import { eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, User, users } from "../drizzle/schema";
import { ENV } from "./_core/env";
import * as mongo from "./mongodb";

let _mysqlDb: ReturnType<typeof drizzle> | null = null;

// Fallback in-memory store for studies when DB is offline during local testing
const _memoryStudies = new Map<string, mongo.MongoStudy>();

export function isMongoActive(): boolean {
  return mongo.isMongoConfigured();
}

// Lazily create the drizzle MySQL instance if a MySQL URL is provided
export async function getMysqlDb() {
  if (!_mysqlDb && process.env.DATABASE_URL) {
    const url = process.env.DATABASE_URL.trim();
    if (!url.startsWith("mongodb://") && !url.startsWith("mongodb+srv://")) {
      try {
        _mysqlDb = drizzle(url);
      } catch (error) {
        console.warn("[MySQL] Failed to connect:", error);
        _mysqlDb = null;
      }
    }
  }
  return _mysqlDb;
}

// ─── USER OPERATIONS ────────────────────────────────────────────────────────

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  // 1. If MongoDB is configured, store in MongoDB
  if (mongo.isMongoConfigured()) {
    try {
      await mongo.upsertMongoUser({
        openId: user.openId,
        name: user.name ?? null,
        email: user.email ?? null,
        loginMethod: user.loginMethod ?? null,
        role: user.role,
        lastSignedIn: user.lastSignedIn ?? new Date(),
      });
      return;
    } catch (err) {
      console.error("[MongoDB] Failed to upsert user:", err);
      // Fall through to try MySQL if available
    }
  }

  // 2. Fall back to MySQL if configured
  const db = await getMysqlDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: No database available (MongoDB/MySQL unset)");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (ENV.ownerOpenId && ENV.ownerOpenId.trim().length > 0 && user.openId === ENV.ownerOpenId) {
      values.role = "admin";
      updateSet.role = "admin";
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[MySQL] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string): Promise<User | undefined> {
  // 1. Try MongoDB
  if (mongo.isMongoConfigured()) {
    try {
      const mUser = await mongo.getMongoUserByOpenId(openId);
      if (mUser) {
        return {
          id: 1,
          openId: mUser.openId,
          name: mUser.name || null,
          email: mUser.email || null,
          loginMethod: mUser.loginMethod || null,
          role: mUser.role,
          createdAt: mUser.createdAt || new Date(),
          updatedAt: mUser.updatedAt || new Date(),
          lastSignedIn: mUser.lastSignedIn || new Date(),
        };
      }
    } catch (err) {
      console.warn("[MongoDB] Error fetching user:", err);
    }
  }

  // 2. Try MySQL
  const db = await getMysqlDb();
  if (db) {
    try {
      const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
      if (result.length > 0) return result[0];
    } catch (err) {
      console.warn("[MySQL] Error fetching user:", err);
    }
  }

  return undefined;
}

// ─── STUDIES OPERATIONS (ULTRASOUND / PREDICTION DATA) ─────────────────────

export async function saveStudy(study: {
  studyId: string;
  userOpenId: string;
  filename: string;
  uri?: string | null;
  probability?: number | null;
  quality?: number | null;
  triage: string;
  notes?: string | null;
  width?: number | null;
  height?: number | null;
  raw?: Record<string, unknown> | null;
}): Promise<{ success: boolean; studyId: string; storedIn: "mongodb" | "memory" }> {
  if (mongo.isMongoConfigured()) {
    const saved = await mongo.saveMongoStudy(study);
    if (saved) {
      return { success: true, studyId: study.studyId, storedIn: "mongodb" };
    }
  }

  // Fallback in-memory persistence when MongoDB connection string is not yet supplied
  const record: mongo.MongoStudy = {
    ...study,
    createdAt: new Date(),
    updatedAt: new Date(),
  };
  _memoryStudies.set(`${study.userOpenId}:${study.studyId}`, record);
  return { success: true, studyId: study.studyId, storedIn: "memory" };
}

export async function listStudies(userOpenId: string): Promise<mongo.MongoStudy[]> {
  if (mongo.isMongoConfigured()) {
    const list = await mongo.listMongoStudies(userOpenId);
    if (list.length > 0 || (await mongo.connectMongo())) {
      return list;
    }
  }

  // In-memory fallback
  const results: mongo.MongoStudy[] = [];
  for (const [key, val] of _memoryStudies.entries()) {
    if (key.startsWith(`${userOpenId}:`)) {
      results.push(val);
    }
  }
  return results.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
}

export async function getStudyById(studyId: string, userOpenId: string): Promise<mongo.MongoStudy | null> {
  if (mongo.isMongoConfigured()) {
    return mongo.getMongoStudyById(studyId, userOpenId);
  }
  return _memoryStudies.get(`${userOpenId}:${studyId}`) || null;
}

export async function deleteStudy(studyId: string, userOpenId: string): Promise<boolean> {
  if (mongo.isMongoConfigured()) {
    return mongo.deleteMongoStudy(studyId, userOpenId);
  }
  return _memoryStudies.delete(`${userOpenId}:${studyId}`);
}

export async function getDatabaseStats() {
  if (mongo.isMongoConfigured()) {
    return mongo.getMongoStats();
  }
  return {
    connected: false,
    databaseType: "In-Memory / Unconnected",
    totalUsers: 0,
    totalStudies: _memoryStudies.size,
    triageCounts: {},
  };
}
