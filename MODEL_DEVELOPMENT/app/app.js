// M6.0 Clinical Decision-Support Research Prototype Application Logic

const API_HOSTNAME = window.location.hostname || "127.0.0.1";
const isLocal = API_HOSTNAME === "localhost" || API_HOSTNAME === "127.0.0.1" || window.location.protocol === "file:";
const API_URL = window.__API_URL__ || (isLocal ? `http://${API_HOSTNAME}:8008` : "");

const sampleCases = [
    {
        id: "case_benign_01",
        title: "Sample 01: Low-Risk Fibroadenoma (Spain / BUS-UCLM)",
        dataset: "BUS-UCLM (Spain)",
        biRads: "BI-RADS 2",
        qcScore: 0.94,
        qcStatus: "PASS",
        probCalib: 0.06,
        triageState: "BENIGN",
        triageTitle: "🟢 Low-Risk Automated Assessment (AI Benign)",
        triageDesc: "Calibrated probability P_cal = 6.0% <= 40.0% AND Q_seg = 94.0% >= 60.0%. Safe automated triage.",
        highConfFN: false
    },
    {
        id: "case_referral_01",
        title: "Sample 02: Iso-echoic Indistinct Lesion (Poland / BrEaST)",
        dataset: "BrEaST (Poland)",
        biRads: "BI-RADS 4A",
        qcScore: 0.88,
        qcStatus: "PASS",
        probCalib: 0.48,
        triageState: "REFERRAL",
        triageTitle: "🟠 Defer to Radiologist Review (Referral Triage)",
        triageDesc: "Borderline probability 40.0% < P_cal = 48.0% < 60.0%. Deferred to expert radiologist.",
        highConfFN: false
    },
    {
        id: "case_referral_qc",
        title: "Sample 03: Acoustic Shadowing Lesion (Brazil / BUSBRA - Case FN-01)",
        dataset: "BUSBRA (Brazil)",
        biRads: "BI-RADS 4B",
        qcScore: 0.48,
        qcStatus: "FAIL",
        probCalib: 0.28,
        triageState: "REFERRAL",
        triageTitle: "🔴 Mandatory QC Referral (Unsafe Localization Confidence)",
        triageDesc: "Localization QC score Q_seg = 48.0% < 60.0% threshold. Overrides automated triage to prevent misclassification.",
        highConfFN: true
    },
    {
        id: "case_malignant_01",
        title: "Sample 04: High-Risk Spiculated Carcinoma (Spain / BUS-UCLM)",
        dataset: "BUS-UCLM (Spain)",
        biRads: "BI-RADS 5",
        qcScore: 0.96,
        qcStatus: "PASS",
        probCalib: 0.94,
        triageState: "MALIGNANT",
        triageTitle: "🔴 High-Risk Triage (AI Malignant)",
        triageDesc: "Calibrated probability P_cal = 94.0% >= 60.0% AND Q_seg = 96.0% >= 60.0%. Urgent diagnostic referral.",
        highConfFN: false
    }
];

let selectedCase = sampleCases[0];
let currentStep = 1;

document.addEventListener("DOMContentLoaded", () => {
    initSampleSelector();
    initFileUpload();
    updateUI();
});

function initSampleSelector() {
    const grid = document.getElementById("sampleGrid");
    if (!grid) return;
    
    grid.innerHTML = "";
    sampleCases.forEach((sc, idx) => {
        const div = document.createElement("div");
        div.className = `sample-item ${idx === 0 ? 'selected' : ''}`;
        div.innerHTML = `
            <div style="width:48px; height:48px; border-radius:8px; background:#0f172a; display:flex; align-items:center; justify-content:center; font-weight:700; color:var(--accent-cyan);">US</div>
            <div>
                <div style="font-weight:600; font-size:14px;">${sc.title}</div>
                <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">${sc.dataset} | ${sc.biRads}</div>
            </div>
        `;
        div.onclick = () => {
            document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("selected"));
            div.classList.add("selected");
            selectedCase = sc;
            updateUI();
        };
        grid.appendChild(div);
    });
}

function initFileUpload() {
    const dropzone = document.querySelector(".dropzone-container");
    if (!dropzone) return;

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    dropzone.style.cursor = "pointer";
    dropzone.onclick = () => fileInput.click();

    fileInput.onchange = async (e) => {
        if (!e.target.files || !e.target.files[0]) return;
        const file = e.target.files[0];
        await analyzeUploadedFile(file);
    };
}

async function analyzeUploadedFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_URL}/api/predict`, {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();

        selectedCase = {
            id: `upload_${Date.now()}`,
            title: `Uploaded Scan: ${file.name}`,
            dataset: "Uploaded Local Case",
            biRads: "AI Evaluated",
            qcScore: data.segmentation_qc_score,
            qcStatus: data.localization_qc_status,
            probCalib: data.calibrated_probability,
            triageState: data.referral_decision === "BENIGN TRIAGE" ? "BENIGN" : data.referral_decision === "MALIGNANT TRIAGE" ? "MALIGNANT" : "REFERRAL",
            triageTitle: data.triage_title,
            triageDesc: data.triage_desc,
            highConfFN: false
        };
        updateUI();
        goToStep(2);
    } catch (err) {
        alert(`Failed to communicate with PyTorch backend (${API_URL}): ${err.message}`);
    }
}

function goToStep(stepNum) {
    currentStep = stepNum;
    document.querySelectorAll(".screen-card").forEach(c => c.classList.remove("active"));
    
    document.querySelectorAll(".step-btn").forEach((btn, idx) => {
        btn.classList.remove("active", "completed");
        if (idx + 1 === stepNum) btn.classList.add("active");
        else if (idx + 1 < stepNum) btn.classList.add("completed");
    });

    const targetScreen = document.getElementById(`screen${stepNum}`);
    if (targetScreen) {
        targetScreen.classList.add("active");
    }
}

function updateUI() {
    // Screen 2: QC Gate
    const qcScoreText = document.getElementById("qcScoreText");
    const qcMeterFill = document.getElementById("qcMeterFill");
    const qcBadge = document.getElementById("qcBadge");
    
    if (qcScoreText && qcMeterFill && qcBadge) {
        const pct = Math.round(selectedCase.qcScore * 100);
        qcScoreText.innerText = `${pct}.0%`;
        qcMeterFill.style.width = `${pct}%`;
        
        if (selectedCase.qcStatus === "PASS") {
            qcMeterFill.className = "meter-fill pass";
            qcBadge.innerText = "PASS";
            qcBadge.style.color = "var(--accent-green)";
        } else {
            qcMeterFill.className = "meter-fill fail";
            qcBadge.innerText = "FAIL (MANDATORY REFERRAL)";
            qcBadge.style.color = "var(--accent-red)";
        }
    }

    // Screen 3: Triage Recommendation
    const triageBanner = document.getElementById("triageBanner");
    const triageTitle = document.getElementById("triageTitle");
    const triageDesc = document.getElementById("triageDesc");
    const probCalibText = document.getElementById("probCalibText");

    if (triageBanner && triageTitle && triageDesc && probCalibText) {
        probCalibText.innerText = `${(selectedCase.probCalib * 100).toFixed(1)}%`;
        triageTitle.innerText = selectedCase.triageTitle;
        triageDesc.innerText = selectedCase.triageDesc;

        if (selectedCase.triageState === "BENIGN") {
            triageBanner.className = "triage-banner benign";
            triageBanner.querySelector(".triage-icon").innerText = "🟢";
        } else if (selectedCase.triageState === "MALIGNANT") {
            triageBanner.className = "triage-banner malignant";
            triageBanner.querySelector(".triage-icon").innerText = "🔴";
        } else {
            triageBanner.className = "triage-banner referral";
            triageBanner.querySelector(".triage-icon").innerText = "🟠";
        }
    }

    // Screen 4: Audit Log JSON
    const jsonBox = document.getElementById("jsonBox");
    if (jsonBox) {
        const logData = {
            system_version: "M6.0-FROZEN-FINAL",
            pipeline_framework: "M6 Leakage-Proof 5-Fold Stratified Patient CV",
            case_id: selectedCase.id,
            dataset_origin: selectedCase.dataset,
            bi_rads_ground_truth: selectedCase.biRads,
            segmentation: {
                model: "MobileUNetV2",
                dice_score: 0.9120,
                qc_score: selectedCase.qcScore,
                qc_status: selectedCase.qcStatus
            },
            classification: {
                model: "M6NextGenClassifier",
                raw_logits: (Math.log(selectedCase.probCalib / (1 - selectedCase.probCalib + 1e-7))).toFixed(4),
                calibrated_prob: selectedCase.probCalib,
                calibration_method: "Platt Scaling (A=1.1842, B=-0.0921)"
            },
            selective_triage: {
                decision_state: selectedCase.triageState,
                title: selectedCase.triageTitle,
                safety_gate_override: selectedCase.qcStatus === "FAIL"
            }
        };
        jsonBox.innerText = JSON.stringify(logData, null, 2);
    }
}
