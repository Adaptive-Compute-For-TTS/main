# Adaptive Compute input design

Keep the existing website's warm stone and ink identity, Geist typography, thin rules, and original logo. The header places the logo and Adaptive Compute wordmark at the top left. Its right side stays empty.

Canvas #e7e4dc, ink #15181b, muted #565c61, rules #cdc8bd, inset surface #e0dcd3, and selected accent #d2cab9. Use flat surfaces without gradients or shadows.

The page contains one freeform text area and an NFE choice of 4, 20, or 32. A large regular-weight heading introduces the controls. Desktop places the text area beside a vertical NFE group; mobile stacks the group beneath the text with three choices across. NFE 20 is selected initially.

Use a visible text label and a fieldset legend for the native radio inputs. Entire NFE options are clickable, selected states are clear, and keyboard focus outlines remain visible. Respect reduced-motion preferences.

Place one dark square arrow button below the text area, aligned to its right edge. Give it an accessible Generate speech label and disable it for blank input or a pending request. Reserve space beside it for concise submitting, success, and error messages; preserve text on failure. The button submits text and the selected NFE to the configured FastAPI service.

The former prompt browser, demo steps, audio comparisons, metrics, recommendation flow, and API-key setup have been removed.

The bottom-right Your runs control opens a full-height native modal sidebar, up to 1120px wide. Focus enters the close button on open and returns to the trigger on dismissal; Escape closes it. The table has clear column headings for text/run ID, NFE, current progress, creation time and local audio. Use flat rows and thin rules, with a keyboard-accessible prompt button in each row. Rows open a dedicated detail view with a back-to-table action. Preserve the actual queued/running/completed/failed states and refresh active runs every two seconds. Do not show fabricated percentages. On narrow screens, prioritize text, NFE and progress, with creation time and audio storage available inside run details; contain any table overflow in a labeled horizontal scroll region. Detail content fits the viewport. Completed runs play locally cached audio with native controls and provide a WAV download. Separate performance, WAV validation, telemetry, quality and full-result sections show real values or truthful missing-data text. Validation distinguishes Passed, Passed with cautions, and Failed, with readable flags and signal checks. Telemetry groups runtime information and NVIDIA readings, explains CPU limitations, and excludes validation/scoring time. Quality shows WER/CER as percentages, the recognized transcript, optional speaker similarity, scorer details and explicit errors/skips. Keep the main input form usable when the panel is closed.
