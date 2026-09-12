# BreastVision Mobile Interface Design

## Product intent

BreastVision is a portrait-oriented retrospective research and decision-support workspace for reviewing breast ultrasound studies. It presents AI assessment as a structured aid for professional review, never as an autonomous diagnosis. The interface is designed for one-handed use, short clinical review sessions, and clear separation between image evidence, model output, quality control, and recommended workflow.

## Screen list

| Screen | Primary content and functionality |
|---|---|
| Overview | Branded header, backend connection state, local research-session summary, latest study preview, quick action to begin analysis, and a safety-oriented disclaimer. |
| New Analysis | Image picker and preview, accepted file guidance, privacy note, live/demo mode switch, upload validation, and Analyze Ultrasound action. |
| Analysis Progress | Current study identifier, seven-stage processing pipeline, animated active stage, cancellation/back affordance, and explicit live inference or demo mode label. |
| Analysis Result | Ultrasound viewer with original/overlay/mask controls, malignancy likelihood, calibration label, localization quality, QC gate, recommended workflow, and referral explanation. |
| Studies | Local-only study history with search, triage filter, sort order, compact thumbnails, likelihood/QC values when available, and delete actions. |
| Study Details | Full saved result with viewer, normalized inference fields, pipeline, multi-scale analysis placeholders only when returned by backend, and technical details. |
| Model Assessment | Retrospective development and external evaluation metrics, clearly separated and labeled, with non-diagnostic interpretation notice. |
| Robustness Assessment | Retrospective robustness characterization for blur, speckle noise, and acoustic shadowing, shown as severity rows and trend charts. |
| System Information | Backend endpoint status, live/demo mode explanation, model pipeline vocabulary, safety thresholds, app version, and privacy statement. |
| Settings | Local-session controls, clear local study history with confirmation, demo-mode preference, and accessibility-oriented display options. |

## Key user flows

### Analyze a new ultrasound

1. User opens **New Analysis** from the tab bar or Overview CTA.
2. User selects a JPG, JPEG, or PNG using the native image picker.
3. App shows the filename, file size, preview, and resolution, and rejects unsupported or oversized files with a recoverable message.
4. User chooses live inference by default, or enables the clearly labeled development-only Demo Data toggle.
5. User taps **Analyze Ultrasound**. The action is disabled while the request is active.
6. App shows **Analysis Progress** and advances the pipeline based on request lifecycle rather than a long artificial delay.
7. On success, the normalized response is persisted as a local research-session study and the user is taken to **Analysis Result**.
8. On failure, the selected image remains available on New Analysis with a concise recovery message.

### Review a result

1. User sees the source ultrasound in a dark viewer surface with a high-contrast bottom control rail.
2. User toggles Original, Overlay, or Mask when segmentation data is available; otherwise the UI explains that lesion localization is unavailable and recommends radiologist review.
3. User reviews calibrated malignancy likelihood and localization quality as separate values.
4. The app applies the safety hierarchy: QC below 0.60 forces Referral Required; otherwise probability at or below 0.15 is Benign Triage, at or above 0.85 is Malignant Triage, and intermediate values defer to Radiologist Review.
5. User can expand technical details and tap pipeline stages to read concise explanations.

### Browse local studies

1. User opens **Studies**.
2. User searches by study ID and filters by workflow state.
3. User taps a study card to open Study Details.
4. User can delete a local research-session record after a confirmation step.

## Layout and interaction rules

The app uses a light clinical canvas with rounded cards, restrained shadows, generous spacing, and a deep navy viewer surface. Primary actions use a clear blue accent. All primary controls use native Pressable feedback and descriptive labels. Every result state includes text and an icon in addition to color, while unavailable backend fields render as “Not available from inference response” rather than invented measurements.

On compact portrait screens, content is vertically stacked, the viewer is full width, assessment follows directly below it, and secondary navigation stays in a bottom tab bar. No horizontal scrolling is permitted. The first viewport prioritizes the study title, connection state, latest image/result, and New Analysis action.

## Color choices

| Role | Color | Use |
|---|---|---|
| Ink navy | `#102A43` | Headings, primary text, clinical authority |
| Clinical blue | `#146EF5` | Primary actions, active navigation, links |
| Ice background | `#F5F8FC` | App canvas |
| White surface | `#FFFFFF` | Cards and form surfaces |
| Viewer navy | `#0B1628` | Ultrasound viewer background |
| Teal success | `#168C78` | Low-likelihood and QC-acceptable states |
| Amber review | `#C47A12` | Intermediate probability and review states |
| Medical red | `#C93C4B` | QC failure, referral required, high-risk indicator |
| Muted blue-gray | `#60758A` | Secondary copy and neutral metadata |
| Hairline border | `#D9E2EC` | Quiet separation between surfaces |

## Safety and privacy language

The app consistently uses “malignancy likelihood,” “AI assessment,” “automated triage,” “radiologist review recommended,” and “decision-support result.” It does not request personally identifiable information and identifies saved records as part of a **Local research session**. The persistent disclaimer states: “This system provides retrospective AI decision support and does not replace professional clinical interpretation.”
