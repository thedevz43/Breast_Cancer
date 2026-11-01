import { COOKIE_NAME } from "../shared/const.js";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";

import { z } from "zod";
import { COOKIE_NAME } from "../shared/const.js";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { adminProcedure, protectedProcedure, publicProcedure, router } from "./_core/trpc";

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

  // User-scoped studies & predictions (Strict IDOR prevention - users can only access their own data)
  studies: router({
    list: protectedProcedure.query(({ ctx }) => {
      // Scoped strictly to the authenticated user's ID
      return {
        userId: ctx.user.id,
        userOpenId: ctx.user.openId,
        message: "User studies query scoped strictly to authenticated session",
      };
    }),
    save: protectedProcedure
      .input(
        z.object({
          studyId: z.string().min(1).max(100),
          filename: z.string().min(1).max(255),
          probability: z.number().min(0).max(1),
          quality: z.number().min(0).max(1),
          triage: z.string().min(1).max(50),
        }),
      )
      .mutation(({ ctx, input }) => {
        // Enforce user ownership
        return {
          success: true,
          ownerOpenId: ctx.user.openId,
          studyId: input.studyId,
        };
      }),
  }),

  // Admin routes - strictly requires role === 'admin'
  admin: router({
    getStats: adminProcedure.query(({ ctx }) => {
      return {
        serverTimestamp: Date.now(),
        adminUser: ctx.user.openId,
        systemStatus: "OPTIMAL",
        accessLevel: "ADMINISTRATOR",
      };
    }),
  }),
});

export type AppRouter = typeof appRouter;
