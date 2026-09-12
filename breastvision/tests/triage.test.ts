import { describe, expect, it } from "vitest";
import { getTriage } from "../lib/triage";

describe("BreastVision triage policy", () => {
  it("forces referral when localization quality is below 0.60", () => {
    expect(getTriage(0.04, 0.59)).toBe("REFERRAL REQUIRED");
  });
  it("returns benign triage for low probability when QC passes", () => {
    expect(getTriage(0.15, 0.6)).toBe("BENIGN TRIAGE");
  });
  it("returns malignant triage for high probability when QC passes", () => {
    expect(getTriage(0.85, 0.92)).toBe("MALIGNANT TRIAGE");
  });
  it("defers intermediate probability to radiologist review", () => {
    expect(getTriage(0.5, 0.9)).toBe("RADIOLOGIST REVIEW");
  });
  it("defers when probability is unavailable", () => {
    expect(getTriage(undefined, 0.9)).toBe("RADIOLOGIST REVIEW");
  });
});
