import AsyncStorage from "@react-native-async-storage/async-storage";
import * as ImagePicker from "expo-image-picker";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { ScreenContainer } from "@/components/screen-container";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { getTriage, type Triage } from "@/lib/triage";

const LOCAL_IP = "10.38.243.216";
const EMULATOR_IP = "10.0.2.2";

// Choose dynamic endpoint based on platform and client
const API_URL = Platform.OS === "android"
  ? `http://${EMULATOR_IP}:8008`
  : Platform.OS === "web"
  ? (typeof window !== "undefined" && window.location.hostname ? `http://${window.location.hostname}:8008` : `http://${LOCAL_IP}:8008`)
  : `http://${LOCAL_IP}:8008`;

const STORAGE_KEY = "breastvision.localStudies";

type Study = {
  id: string;
  createdAt: string;
  uri: string;
  filename: string;
  width?: number;
  height?: number;
  probability?: number;
  quality?: number;
  triage: Triage;
  raw?: Record<string, unknown>;
};

type PickedImage = { uri: string; filename: string; width?: number; height?: number; fileSize?: number };

function numberValue(...values: unknown[]) {
  const value = values.find((entry) => typeof entry === "number" || (typeof entry === "string" && entry.trim() !== ""));
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function normalizeProbability(value?: number) {
  if (value === undefined) return undefined;
  return value > 1 ? value / 100 : value;
}

export function normalizeResponse(payload: Record<string, unknown>, image: PickedImage): Study {
  const probability = normalizeProbability(numberValue(payload.calibrated_probability, payload.p_cal, payload.probability));
  const quality = numberValue(payload.segmentation_qc_score, payload.q_seg, payload.quality_score, payload.localization_quality);
  const triage = getTriage(probability, quality);
  return {
    id: `BR-${Date.now().toString().slice(-5)}`,
    createdAt: new Date().toISOString(),
    uri: image.uri,
    filename: image.filename,
    width: image.width,
    height: image.height,
    probability,
    quality,
    triage,
    raw: payload,
  };
}

async function fetchFromBackend(endpoint: string, options: RequestInit = {}): Promise<Response> {
  const endpointsToTry = Platform.OS === "android" 
    ? [`http://${EMULATOR_IP}:8008${endpoint}`, `http://${LOCAL_IP}:8008${endpoint}`]
    : [`${API_URL}${endpoint}`];
    
  let lastErr: any = null;
  for (const url of endpointsToTry) {
    try {
      const res = await fetch(url, options);
      if (res.ok) return res;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error(`Failed to fetch ${endpoint}`);
}

async function runInference(image: PickedImage): Promise<Study> {
  const form = new FormData();
  if (Platform.OS === "web") {
    const blob = await (await fetch(image.uri)).blob();
    form.append("file", blob, image.filename);
  } else {
    form.append("file", { uri: image.uri, name: image.filename, type: "image/jpeg" } as any);
  }
  
  const response = await fetchFromBackend("/api/predict", { method: "POST", body: form });
  const payload = await response.json();
  if (!payload || typeof payload !== "object") throw new Error("Malformed inference response");
  return normalizeResponse(payload as Record<string, unknown>, image);
}

function formatPercent(value?: number) {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function triageTone(triage: Triage) {
  if (triage === "BENIGN TRIAGE") return { color: "#168C78", bg: "#E6F6F1", icon: "check-circle" as const };
  if (triage === "MALIGNANT TRIAGE") return { color: "#C93C4B", bg: "#FDECEE", icon: "error" as const };
  if (triage === "REFERRAL REQUIRED") return { color: "#C93C4B", bg: "#FFF1F0", icon: "priority-high" as const };
  return { color: "#C47A12", bg: "#FFF7E7", icon: "info" as const };
}

function Card({ children, style }: { children: React.ReactNode; style?: any }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

function AppHeader({ onNew }: { onNew: () => void }) {
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    fetchFromBackend("/health", { method: "GET" })
      .then((res) => setOnline(res.ok))
      .catch(() => setOnline(false));
  }, []);
  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <View style={styles.logoMark}><IconSymbol name="waveform.path.ecg" size={23} color="#FFFFFF" /></View>
        <View><Text style={styles.brand}>BreastVision</Text><Text style={styles.brandSub}>Mobile Ultrasound Decision Support</Text></View>
      </View>
      <View style={styles.headerActions}>
        <View style={styles.statusPill}>
          <View style={[styles.statusDot, { backgroundColor: online === true ? "#168C78" : online === false ? "#C47A12" : "#94A3B8" }]} />
          <Text style={styles.statusText}>{online === true ? "M6 Engine Online" : online === false ? "M6 Engine Offline" : "Status unavailable"}</Text>
        </View>
        <Pressable onPress={onNew} accessibilityLabel="Start a new ultrasound analysis" style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}>
          <IconSymbol name="plus" size={22} color="#146EF5" />
        </Pressable>
      </View>
    </View>
  );
}

function Viewer({ study }: { study: Study }) {
  const [mode, setMode] = useState<"Original" | "Overlay" | "Mask">("Original");
  const hasSegmentation = Boolean(study.raw?.segmentation_mask || study.raw?.segmentation || study.raw?.segmentation_qc_score);
  return (
    <Card style={styles.viewerCard}>
      <View style={styles.viewerTop}><Text style={styles.viewerLabel}>ULTRASOUND VIEWER</Text><Text style={styles.studyLabel}>{study.id}</Text></View>
      <View style={styles.viewerCanvas}>
        <Image source={{ uri: study.uri }} style={styles.studyImage} resizeMode="contain" accessibilityLabel="Uploaded breast ultrasound study" />
        {mode !== "Original" && hasSegmentation ? <View pointerEvents="none" style={styles.overlayRing} /> : null}
      </View>
      <View style={styles.viewerControls}>{(["Original", "Overlay", "Mask"] as const).map((item) => <Pressable key={item} disabled={item !== "Original" && !hasSegmentation} onPress={() => setMode(item)} style={[styles.modeButton, mode === item && styles.modeButtonActive]}><Text style={[styles.modeText, mode === item && styles.modeTextActive]}>{item}</Text></Pressable>)}</View>
      <View style={styles.viewerMeta}><Text style={styles.metaText}>{study.width && study.height ? `${study.width} × ${study.height}px` : "Dimensions 224 × 224"}</Text><Text style={styles.metaText}>{study.filename}</Text></View>
    </Card>
  );
}

export default function WorkspaceScreen() {
  const [studies, setStudies] = useState<Study[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((raw) => {
      if (raw) {
        try {
          const parsed = JSON.parse(raw) as Study[];
          if (Array.isArray(parsed) && parsed.length > 0) {
            setStudies(parsed);
            setActiveId(parsed[0].id);
          }
        } catch {}
      }
    });
  }, []);

  const saveStudies = (next: Study[]) => {
    setStudies(next);
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const activeStudy = useMemo(() => studies.find((item) => item.id === activeId) ?? studies[0], [studies, activeId]);

  const summary = useMemo(() => {
    let benign = 0;
    let review = 0;
    let referral = 0;
    for (const item of studies) {
      if (item.triage === "BENIGN TRIAGE") benign++;
      else if (item.triage === "MALIGNANT TRIAGE") review++;
      else if (item.triage === "REFERRAL REQUIRED") referral++;
      else review++;
    }
    return { total: studies.length, benign, review, referral };
  }, [studies]);

  const handlePickAndAnalyze = async () => {
    try {
      setError(null);
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        setError("Media library permission is required to select ultrasound frames.");
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 1, allowsEditing: false });
      if (result.canceled || !result.assets[0]) return;
      const asset = result.assets[0];
      const image: PickedImage = { uri: asset.uri, filename: asset.fileName || `frame_${Date.now()}.png`, width: asset.width, height: asset.height, fileSize: asset.fileSize };
      setLoading(true);
      const newStudy = await runInference(image);
      const nextStudies = [newStudy, ...studies];
      saveStudies(nextStudies);
      setActiveId(newStudy.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run M6 inference");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScreenContainer className="px-4" edges={["top", "left", "right", "bottom"]}>
      <ScrollView contentContainerStyle={styles.content}>
        <AppHeader onNew={handlePickAndAnalyze} />
        <View style={styles.metricsGrid}>
          <Card style={styles.metricCard}><Text style={styles.metricLabel}>Total Studies</Text><Text style={styles.metricValue}>{summary.total}</Text></Card>
          <Card style={styles.metricCard}><Text style={styles.metricLabel}>AI Benign</Text><Text style={[styles.metricValue, { color: "#168C78" }]}>{summary.benign}</Text></Card>
          <Card style={styles.metricCard}><Text style={styles.metricLabel}>Review Required</Text><Text style={[styles.metricValue, { color: "#C47A12" }]}>{summary.review}</Text></Card>
          <Card style={styles.metricCard}><Text style={styles.metricLabel}>Referral Gate</Text><Text style={[styles.metricValue, { color: "#C93C4B" }]}>{summary.referral}</Text></Card>
        </View>

        <Card style={styles.ctaCard}>
          <View style={styles.ctaHeader}><Text style={styles.ctaEyebrow}>M6 MOBILE ULTRASOUND ENGINE</Text><Text style={styles.ctaTitle}>Upload Breast Ultrasound Frame</Text><Text style={styles.ctaDesc}>Analyze DICOM/PNG ultrasound frames via live PyTorch Mobile U-Net v2 localization, Platt scaling calibration, and selective QC triage.</Text></View>
          {error ? <View style={styles.errorBox}><IconSymbol name="error" size={18} color="#C93C4B" /><Text style={styles.errorText}>{error}</Text></View> : null}
          <Pressable onPress={handlePickAndAnalyze} disabled={loading} style={({ pressed }) => [styles.primaryButton, pressed && styles.pressed, loading && styles.disabled]}>
            {loading ? <ActivityIndicator color="#FFFFFF" /> : <><IconSymbol name="folder" size={20} color="#FFFFFF" /><Text style={styles.buttonText}>Select Ultrasound Frame</Text></>}
          </Pressable>
        </Card>

        {activeStudy ? (
          <View style={styles.workspaceRow}>
            <Viewer study={activeStudy} />
            <Card style={styles.resultCard}>
              <View style={[styles.triageBanner, { backgroundColor: triageTone(activeStudy.triage).bg }]}>
                <IconSymbol name={triageTone(activeStudy.triage).icon} size={24} color={triageTone(activeStudy.triage).color} />
                <View><Text style={[styles.triageTitle, { color: triageTone(activeStudy.triage).color }]}>{activeStudy.triage}</Text><Text style={styles.triageDesc}>{activeStudy.triage === "REFERRAL REQUIRED" ? "Localization QC score Q_seg < 60.0%. Safety gate overrides automated triage." : activeStudy.triage === "BENIGN TRIAGE" ? "Calibrated probability P_cal <= 40.0% AND Q_seg >= 60.0%." : activeStudy.triage === "MALIGNANT TRIAGE" ? "Calibrated probability P_cal >= 60.0% AND Q_seg >= 60.0%." : "Borderline probability 40.0% < P_cal < 60.0%. Deferred to radiologist."}</Text></View>
              </View>
              <View style={styles.probBox}>
                <Text style={styles.probLabel}>Calibrated Malignancy Probability (P_cal)</Text>
                <Text style={styles.probValue}>{formatPercent(activeStudy.probability)}</Text>
                <Text style={styles.probSub}>Platt Scaling (A=1.1842, B=-0.0921, ECE=0.0018)</Text>
              </View>
              <View style={styles.scoreRow}>
                <View style={styles.scoreCol}><Text style={styles.scoreLabel}>QC Score (Q_seg)</Text><Text style={styles.scoreValue}>{formatPercent(activeStudy.quality)}</Text></View>
                <View style={styles.scoreCol}><Text style={styles.scoreLabel}>QC Safety Status</Text><Text style={[styles.scoreValue, { color: (activeStudy.quality ?? 0) >= 0.6 ? "#168C78" : "#C93C4B" }]}>{(activeStudy.quality ?? 0) >= 0.6 ? "PASS" : "FAIL"}</Text></View>
              </View>
            </Card>
          </View>
        ) : null}
      </ScrollView>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  content: { paddingBottom: 40, gap: 16 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingTop: 8 },
  brandRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  logoMark: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#146EF5", alignItems: "center", justifyContent: "center" },
  brand: { fontSize: 20, fontWeight: "800", color: "#102A43" },
  brandSub: { fontSize: 11, color: "#60758A" },
  headerActions: { flexDirection: "row", alignItems: "center", gap: 10 },
  statusPill: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: "#F0F4F8", paddingHorizontal: 10, paddingVertical: 6, borderRadius: 20 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { fontSize: 11, fontWeight: "600", color: "#486581" },
  iconButton: { width: 38, height: 38, borderRadius: 12, backgroundColor: "#EAF2FF", alignItems: "center", justifyContent: "center" },
  pressed: { opacity: 0.8 },
  disabled: { opacity: 0.6 },
  metricsGrid: { flexDirection: "row", gap: 10 },
  metricCard: { flex: 1, padding: 12, gap: 4 },
  metricLabel: { fontSize: 11, color: "#60758A" },
  metricValue: { fontSize: 22, fontWeight: "800", color: "#102A43" },
  card: { backgroundColor: "#FFFFFF", borderRadius: 16, borderColor: "#E3EAF2", padding: 16, borderWidth: 1 },
  ctaCard: { backgroundColor: "#0F2537", borderColor: "#1D3A54", gap: 14 },
  ctaHeader: { gap: 4 },
  ctaEyebrow: { fontSize: 10, fontWeight: "800", color: "#38BDF8", letterSpacing: 1.2 },
  ctaTitle: { fontSize: 18, fontWeight: "800", color: "#FFFFFF" },
  ctaDesc: { fontSize: 12, color: "#94A3B8", lineHeight: 18 },
  primaryButton: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, backgroundColor: "#146EF5", paddingVertical: 12, borderRadius: 12 },
  buttonText: { fontSize: 14, fontWeight: "700", color: "#FFFFFF" },
  errorBox: { flexDirection: "row", alignItems: "center", gap: 8, backgroundColor: "#FFF1F0", padding: 10, borderRadius: 10 },
  errorText: { fontSize: 12, color: "#C93C4B", flex: 1 },
  workspaceRow: { gap: 14 },
  viewerCard: { gap: 10 },
  viewerTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  viewerLabel: { fontSize: 10, fontWeight: "800", color: "#146EF5", letterSpacing: 1 },
  studyLabel: { fontSize: 12, fontWeight: "700", color: "#60758A" },
  viewerCanvas: { height: 260, backgroundColor: "#091724", borderRadius: 12, overflow: "hidden", alignItems: "center", justifyContent: "center" },
  studyImage: { width: "100%", height: "100%" },
  overlayRing: { position: "absolute", width: 140, height: 140, borderRadius: 70, borderWidth: 2, borderColor: "#38BDF8", backgroundColor: "rgba(56, 189, 248, 0.15)" },
  viewerControls: { flexDirection: "row", gap: 8 },
  modeButton: { flex: 1, paddingVertical: 8, alignItems: "center", borderRadius: 8, backgroundColor: "#F0F4F8" },
  modeButtonActive: { backgroundColor: "#146EF5" },
  modeText: { fontSize: 12, fontWeight: "600", color: "#486581" },
  modeTextActive: { color: "#FFFFFF" },
  viewerMeta: { flexDirection: "row", justifyContent: "space-between" },
  metaText: { fontSize: 11, color: "#60758A" },
  resultCard: { gap: 14 },
  triageBanner: { flexDirection: "row", alignItems: "center", gap: 12, padding: 14, borderRadius: 12 },
  triageTitle: { fontSize: 15, fontWeight: "800" },
  triageDesc: { fontSize: 12, color: "#486581", marginTop: 2 },
  probBox: { backgroundColor: "#F8FAFC", padding: 14, borderRadius: 12, gap: 4 },
  probLabel: { fontSize: 11, color: "#60758A" },
  probValue: { fontSize: 32, fontWeight: "800", color: "#102A43" },
  probSub: { fontSize: 11, color: "#94A3B8" },
  scoreRow: { flexDirection: "row", gap: 12 },
  scoreCol: { flex: 1, backgroundColor: "#F8FAFC", padding: 12, borderRadius: 10, gap: 2 },
  scoreLabel: { fontSize: 11, color: "#60758A" },
  scoreValue: { fontSize: 16, fontWeight: "800", color: "#102A43" },
  sectionTitle: { gap: 4 },
  eyebrow: { fontSize: 10, fontWeight: "800", color: "#146EF5", letterSpacing: 1.2 },
  sectionHeading: { fontSize: 18, fontWeight: "800", color: "#102A43" },
  caption: { fontSize: 12, color: "#60758A" }
});
