import { z } from "zod";
import { COOKIE_NAME } from "../shared/const.js";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { adminProcedure, protectedProcedure, publicProcedure, router } from "./_core/trpc";
import * as db from "./db";

export const appRouter = router({
  // System health & telemetry
  system: systemRouter,

  // Authentication endpoints
  auth: router({
    me: publicProcedure.query((opts) => opts.ctx.user),
    profile: protectedProcedure.query(({ ctx }) => ({
      id: ctx.user.id,
      openId: ctx.user.openId,
      name: ctx.user.name,
      email: ctx.user.email,
      role: ctx.user.role,
    })),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // User-scoped studies & predictions (MongoDB backed, strict IDOR prevention)
  studies: router({
    list: protectedProcedure.query(async ({ ctx }) => {
      const studies = await db.listStudies(ctx.user.openId);
      return {
        userId: ctx.user.id,
        userOpenId: ctx.user.openId,
        count: studies.length,
        studies,
      };
    }),

    save: protectedProcedure
      .input(
        z.object({
          studyId: z.string().min(1).max(100),
          filename: z.string().min(1).max(255),
          uri: z.string().optional().nullable(),
          probability: z.number().min(0).max(1).optional().nullable(),
          quality: z.number().min(0).max(1).optional().nullable(),
          triage: z.string().min(1).max(50),
          notes: z.string().max(2000).optional().nullable(),
          width: z.number().optional().nullable(),
          height: z.number().optional().nullable(),
          raw: z.record(z.unknown()).optional().nullable(),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        const result = await db.saveStudy({
          studyId: input.studyId,
          userOpenId: ctx.user.openId,
          filename: input.filename,
          uri: input.uri ?? null,
          probability: input.probability ?? null,
          quality: input.quality ?? null,
          triage: input.triage,
          notes: input.notes ?? null,
          width: input.width ?? null,
          height: input.height ?? null,
          raw: input.raw ?? null,
        });

        return {
          success: result.success,
          studyId: result.studyId,
          storedIn: result.storedIn,
          ownerOpenId: ctx.user.openId,
        };
      }),

    getById: protectedProcedure
      .input(z.object({ studyId: z.string().min(1) }))
      .query(async ({ ctx, input }) => {
        const study = await db.getStudyById(input.studyId, ctx.user.openId);
        return { study };
      }),

    delete: protectedProcedure
      .input(z.object({ studyId: z.string().min(1) }))
      .mutation(async ({ ctx, input }) => {
        const deleted = await db.deleteStudy(input.studyId, ctx.user.openId);
        return { success: deleted, studyId: input.studyId };
      }),
  }),

  // Admin routes - strictly requires role === 'admin'
  admin: router({
    getStats: adminProcedure.query(async ({ ctx }) => {
      const dbStats = await db.getDatabaseStats();
      return {
        serverTimestamp: Date.now(),
        adminUser: ctx.user.openId,
        systemStatus: "OPTIMAL",
        accessLevel: "ADMINISTRATOR",
        database: dbStats,
      };
    }),
  }),
});

export type AppRouter = typeof appRouter;
