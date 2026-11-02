import { MongoClient, Db, Collection, IndexDescription } from "mongodb";
import { ENV } from "./_core/env";

export interface MongoUser {
  openId: string;
  name?: string | null;
  email?: string | null;
  loginMethod?: string | null;
  role: "user" | "admin";
  createdAt: Date;
  updatedAt: Date;
  lastSignedIn: Date;
}

export interface MongoStudy {
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
  createdAt: Date;
  updatedAt: Date;
}

let client: MongoClient | null = null;
let db: Db | null = null;
let isConnecting = false;
let indicesCreated = false;

export function getMongoUri(): string {
  if (ENV.mongodbUri && ENV.mongodbUri.trim().length > 0) {
    return ENV.mongodbUri.trim();
  }
  const dbUrl = ENV.databaseUrl.trim();
  if (dbUrl.startsWith("mongodb://") || dbUrl.startsWith("mongodb+srv://")) {
    return dbUrl;
  }
  return "";
}

export function isMongoConfigured(): boolean {
  return getMongoUri().length > 0;
}

export async function connectMongo(): Promise<Db | null> {
  if (db) return db;
  const uri = getMongoUri();
  if (!uri) {
    return null;
  }

  if (isConnecting) {
    // Wait briefly for connection
    await new Promise((r) => setTimeout(r, 200));
    if (db) return db;
  }

  try {
    isConnecting = true;
    client = new MongoClient(uri, {
      maxPoolSize: 10,
      minPoolSize: 1,
      serverSelectionTimeoutMS: 5000,
      connectTimeoutMS: 10000,
    });

    await client.connect();
    // Default database name if not specified in URI is 'breastvision'
    db = client.db(process.env.MONGODB_DB_NAME || "breastvision");
    console.log(`[MongoDB] Connected successfully to database: ${db.databaseName}`);

    if (!indicesCreated) {
      await ensureIndices(db);
      indicesCreated = true;
    }

    return db;
  } catch (error) {
    console.error("[MongoDB] Connection failed:", error);
    db = null;
    client = null;
    return null;
  } finally {
    isConnecting = false;
  }
}

async function ensureIndices(database: Db) {
  try {
    const usersCol = database.collection<MongoUser>("users");
    await usersCol.createIndex({ openId: 1 }, { unique: true });

    const studiesCol = database.collection<MongoStudy>("studies");
    await studiesCol.createIndex({ studyId: 1 }, { unique: true });
    await studiesCol.createIndex({ userOpenId: 1, createdAt: -1 });
    console.log("[MongoDB] Database indices verified");
  } catch (err) {
    console.warn("[MongoDB] Index creation warning:", err);
  }
}

export async function getUsersCollection(): Promise<Collection<MongoUser> | null> {
  const database = await connectMongo();
  return database ? database.collection<MongoUser>("users") : null;
}

export async function getStudiesCollection(): Promise<Collection<MongoStudy> | null> {
  const database = await connectMongo();
  return database ? database.collection<MongoStudy>("studies") : null;
}

// ─── USER OPERATIONS ────────────────────────────────────────────────────────

export async function upsertMongoUser(userData: {
  openId: string;
  name?: string | null;
  email?: string | null;
  loginMethod?: string | null;
  role?: "user" | "admin";
  lastSignedIn?: Date;
}): Promise<MongoUser | null> {
  const col = await getUsersCollection();
  if (!col) return null;

  const now = new Date();
  let role: "user" | "admin" = userData.role || "user";
  if (ENV.ownerOpenId && ENV.ownerOpenId.trim() && userData.openId === ENV.ownerOpenId.trim()) {
    role = "admin";
  }

  const updateDoc: Partial<MongoUser> = {
    updatedAt: now,
    lastSignedIn: userData.lastSignedIn || now,
    role,
  };

  if (userData.name !== undefined) updateDoc.name = userData.name;
  if (userData.email !== undefined) updateDoc.email = userData.email;
  if (userData.loginMethod !== undefined) updateDoc.loginMethod = userData.loginMethod;

  const result = await col.findOneAndUpdate(
    { openId: userData.openId },
    {
      $set: updateDoc,
      $setOnInsert: {
        openId: userData.openId,
        createdAt: now,
      },
    },
    { upsert: true, returnDocument: "after" },
  );

  return result || null;
}

export async function getMongoUserByOpenId(openId: string): Promise<MongoUser | null> {
  const col = await getUsersCollection();
  if (!col) return null;
  return col.findOne({ openId });
}

// ─── STUDY OPERATIONS ───────────────────────────────────────────────────────

export async function saveMongoStudy(studyData: {
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
}): Promise<MongoStudy | null> {
  const col = await getStudiesCollection();
  if (!col) return null;

  const now = new Date();
  const updateData: Partial<MongoStudy> = {
    filename: studyData.filename,
    triage: studyData.triage,
    updatedAt: now,
  };

  if (studyData.uri !== undefined) updateData.uri = studyData.uri;
  if (studyData.probability !== undefined) updateData.probability = studyData.probability;
  if (studyData.quality !== undefined) updateData.quality = studyData.quality;
  if (studyData.notes !== undefined) updateData.notes = studyData.notes;
  if (studyData.width !== undefined) updateData.width = studyData.width;
  if (studyData.height !== undefined) updateData.height = studyData.height;
  if (studyData.raw !== undefined) updateData.raw = studyData.raw;

  const result = await col.findOneAndUpdate(
    { studyId: studyData.studyId, userOpenId: studyData.userOpenId },
    {
      $set: updateData,
      $setOnInsert: {
        studyId: studyData.studyId,
        userOpenId: studyData.userOpenId,
        createdAt: now,
      },
    },
    { upsert: true, returnDocument: "after" },
  );

  return result || null;
}

export async function listMongoStudies(userOpenId: string, limit: number = 100): Promise<MongoStudy[]> {
  const col = await getStudiesCollection();
  if (!col) return [];
  return col
    .find({ userOpenId })
    .sort({ createdAt: -1 })
    .limit(limit)
    .toArray();
}

export async function getMongoStudyById(studyId: string, userOpenId: string): Promise<MongoStudy | null> {
  const col = await getStudiesCollection();
  if (!col) return null;
  return col.findOne({ studyId, userOpenId });
}

export async function deleteMongoStudy(studyId: string, userOpenId: string): Promise<boolean> {
  const col = await getStudiesCollection();
  if (!col) return false;
  const res = await col.deleteOne({ studyId, userOpenId });
  return (res.deletedCount || 0) > 0;
}

export async function getMongoStats() {
  const database = await connectMongo();
  if (!database) {
    return {
      connected: false,
      databaseType: "MongoDB (Not Connected)",
      totalUsers: 0,
      totalStudies: 0,
      triageCounts: {},
    };
  }

  const [totalUsers, totalStudies, triageAgg] = await Promise.all([
    database.collection("users").countDocuments(),
    database.collection("studies").countDocuments(),
    database
      .collection("studies")
      .aggregate([{ $group: { _id: "$triage", count: { $sum: 1 } } }])
      .toArray(),
  ]);

  const triageCounts: Record<string, number> = {};
  for (const item of triageAgg) {
    if (item._id) triageCounts[String(item._id)] = item.count;
  }

  return {
    connected: true,
    databaseType: "MongoDB",
    databaseName: database.databaseName,
    totalUsers,
    totalStudies,
    triageCounts,
  };
}

