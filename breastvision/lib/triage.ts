export type Triage = "BENIGN TRIAGE" | "MALIGNANT TRIAGE" | "RADIOLOGIST REVIEW" | "REFERRAL REQUIRED";

export function getTriage(probability?: number, quality?: number): Triage {
  if (quality !== undefined && quality < 0.6) return "REFERRAL REQUIRED";
  if (probability === undefined) return "RADIOLOGIST REVIEW";
  if (probability <= 0.40) return "BENIGN TRIAGE";
  if (probability >= 0.60) return "MALIGNANT TRIAGE";
  return "RADIOLOGIST REVIEW";
}
