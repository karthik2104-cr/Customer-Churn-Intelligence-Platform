import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Copy, RefreshCw } from "lucide-react";
import { ErrorBlock, LoadingBlock, PageHeader, Panel, RiskBadge, buttonClass, controlClass, ghostButtonClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, get, patch, post } from "../services/api";
import type { Campaign, Factor, Prediction, Recommendation } from "../types";
import { compactDate, money, number, percent, titleCase } from "../utils/format";

interface Detail {
  profile: Record<string, string | number | null>;
  profile_note: string;
  prediction: Prediction | null;
  history: Prediction[];
  campaigns: Campaign[];
  explanation_limitation: string;
  decision_support_warning: string;
}

const ACCOUNT = ["Contract", "tenure", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"] as const;
const SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"] as const;
const DESCRIPTIVE = ["Partner", "Dependents", "gender", "SeniorCitizen"] as const;
const STATUSES = ["follow_up_required", "contacted", "offer_proposed", "retained"];

const LABELS: Record<string, string> = {
  Contract: "Contract",
  tenure: "Tenure",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
  Churn: "Observed churn",
  PhoneService: "Phone service",
  MultipleLines: "Multiple lines",
  InternetService: "Internet",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Tech support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Partner: "Partner",
  Dependents: "Dependents",
  gender: "Gender",
  SeniorCitizen: "Senior citizen",
};

function displayValue(key: string, value: string | number | null | undefined): string {
  if (value == null || value === "" || (typeof value === "number" && Number.isNaN(value))) return "n/a";
  if (key === "MonthlyCharges" || key === "TotalCharges") return money(Number(value), 2);
  if (key === "tenure") return `${value} ${Number(value) === 1 ? "month" : "months"}`;
  if (key === "SeniorCitizen") return Number(value) === 1 ? "Yes" : "No";
  return String(value);
}

function directionLabel(direction: string): string {
  if (direction === "higher_predicted_churn") return "Associated with a higher predicted probability";
  if (direction === "lower_predicted_churn") return "Associated with a lower predicted probability";
  return "No score change versus the baseline";
}

export function CustomerPage() {
  const { customerId = "" } = useParams();
  const state = useApi<Detail>(customerId ? `/api/customers/${encodeURIComponent(customerId)}` : null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notes, setNotes] = useState<Record<number, string>>({});

  async function refreshScore() {
    setBusy(true);
    setError(null);
    try {
      await get<Detail>(`/api/customers/${encodeURIComponent(customerId)}?refresh_explanation=true`);
      setMessage("Prediction refreshed from the loaded model.");
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "The score could not be refreshed.");
    } finally {
      setBusy(false);
    }
  }

  async function createFollowUp(recommendation: Recommendation, priority: number | null) {
    setError(null);
    try {
      await post("/api/campaigns", {
        customer_id: customerId,
        action: recommendation.action,
        reason: recommendation.reason,
        effort: recommendation.effort,
        category: recommendation.category,
        status: "follow_up_required",
        priority_score: priority || 0,
      });
      setMessage("Follow-up recorded. This does not mean the customer was retained.");
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "The follow-up could not be saved.");
    }
  }

  async function updateCampaign(campaign: Campaign, status: string) {
    if (status === "retained" && !window.confirm("Mark this follow-up retained? The dataset has no outcome field, so this only records your status.")) return;
    setError(null);
    try {
      await patch(`/api/campaigns/${campaign.id}`, { status, notes: notes[campaign.id] ?? campaign.notes ?? "" });
      setMessage("Follow-up status saved.");
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "The status could not be saved.");
    }
  }

  async function copyId() {
    try {
      await navigator.clipboard.writeText(customerId);
      setMessage("Customer id copied.");
    } catch {
      setError("The browser did not allow copying. Select the id and copy it manually.");
    }
  }

  if (state.loading) return <LoadingBlock label="Loading customer profile" />;
  if (state.error) return <ErrorBlock message={state.error} />;
  if (!state.data) return null;
  const { profile, prediction } = state.data;
  const probability = prediction?.probability ?? 0;
  const gauge = Math.max(0, Math.min(100, probability * 100));
  const high = prediction?.thresholds?.high;
  const medium = prediction?.thresholds?.medium;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Link className={ghostButtonClass} to="/customers"><ArrowLeft size={16} aria-hidden="true" /> Customer profiles</Link>
        <Link className={ghostButtonClass} to="/explain">Explainability</Link>
        <Link className={ghostButtonClass} to="/campaigns">Campaigns</Link>
        <button type="button" className={ghostButtonClass} onClick={copyId}><Copy size={16} aria-hidden="true" /> Copy id</button>
        <button type="button" className={buttonClass} disabled={busy} onClick={refreshScore}>
          <RefreshCw size={16} aria-hidden="true" /> {busy ? "Refreshing…" : "Refresh prediction"}
        </button>
      </div>
      <PageHeader eyebrow="02 — Customer profile" title={String(profile.customerID || customerId)} lede="The account, the current model score, the factors behind that score, and the follow-up history for this customer." />
      {message && <p className="mb-3 rounded-lg border border-good/30 bg-good/10 px-3 py-2 text-sm text-good" role="status">{message}</p>}
      {error && <div className="mb-3"><ErrorBlock message={error} /></div>}

      <section className="grid gap-4 lg:grid-cols-5">
        <div className="rounded-2xl border border-line bg-surface p-5 lg:col-span-3">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">Subscription</p>
          <h2 className="mt-2 font-serif text-3xl">{displayValue("Contract", profile.Contract)}</h2>
          <p className="mt-1 text-sm text-muted">{displayValue("tenure", profile.tenure)} · {displayValue("PaymentMethod", profile.PaymentMethod)}</p>
          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
            {ACCOUNT.filter((key) => key !== "Contract" && key !== "tenure" && key !== "PaymentMethod").map((key) => (
              <div key={key}>
                <dt className="text-muted">{LABELS[key]}</dt>
                <dd className="font-semibold">{displayValue(key, profile[key])}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-5 flex flex-wrap gap-2">
            {SERVICES.map((key) => (
              <span key={key} className="rounded-md border border-line bg-canvas px-2 py-1 text-xs">
                <span className="text-muted">{LABELS[key]}: </span>{displayValue(key, profile[key])}
              </span>
            ))}
          </div>
          <p className="mt-4 text-xs leading-5 text-muted">{state.data.profile_note}</p>
          <dl className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            {DESCRIPTIVE.map((key) => (
              <div key={key}>
                <dt className="text-muted">{LABELS[key]}</dt>
                <dd>{displayValue(key, profile[key])}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="rounded-2xl border border-line bg-[var(--sidebar)] p-5 text-white lg:col-span-2">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-white/60">Predicted churn probability</p>
          {prediction ? (
            <>
              <div className="mt-4 flex items-center gap-5">
                <div
                  className="grid h-32 w-32 shrink-0 place-items-center rounded-full"
                  style={{ background: `conic-gradient(rgb(var(--accent)) ${gauge}%, rgb(255 255 255 / 0.16) 0)` }}
                  role="img"
                  aria-label={`Predicted churn probability ${percent(prediction.probability)}`}
                >
                  <div className="grid h-24 w-24 place-items-center rounded-full bg-[var(--sidebar)]">
                    <span className="font-serif text-2xl">{percent(prediction.probability)}</span>
                  </div>
                </div>
                <div>
                  <RiskBadge risk={prediction.risk_category} />
                  <p className="mt-3 text-sm text-white/80">Predicted class {prediction.predicted_class || "n/a"}</p>
                  <p className="text-sm text-white/80">Priority {number(prediction.priority_score, 1)}</p>
                </div>
              </div>
              <p className="mt-4 text-sm leading-6 text-white/80">
                {prediction.risk_category === "High"
                  ? `This score is in the high-risk band because it is at or above ${percent(high, 0)}. It is a model probability, not a guarantee that the customer will leave.`
                  : prediction.risk_category === "Medium"
                    ? `This score is elevated, between the medium cutoff ${percent(medium, 0)} and the decision threshold ${percent(high, 0)}. It is not a guarantee of churn.`
                    : `This score is below the medium cutoff of ${percent(medium, 0)}. A low score is still a probability, not proof the customer will stay.`}
              </p>
              <p className="mt-2 text-xs text-white/60">Scored {compactDate(prediction.created_at)} · {prediction.explanation_method || "explanation unavailable"}</p>
              <p className="mt-3 border-t border-white/15 pt-3 text-sm text-white/75">Observed label in the source file: {displayValue("Churn", profile.Churn)}. That label is separate from the prediction.</p>
            </>
          ) : (
            <p className="mt-4 text-sm text-white/75">No score is stored yet. Load the model, then refresh the prediction.</p>
          )}
        </div>
      </section>

      <div className="mt-4 grid gap-4 lg:grid-cols-5">
        <Panel className="lg:col-span-3" title="Why this score" description={state.data.explanation_limitation}>
            {prediction?.factors?.length ? (
              <ul className="space-y-3">
                {prediction.factors.map((factor: Factor) => (
                  <li key={`${factor.feature}-${factor.detail}`}>
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="font-semibold">{factor.feature}</span>
                      <span className={factor.contribution > 0 ? "text-bad" : factor.contribution < 0 ? "text-good" : "text-muted"}>
                        {factor.contribution > 0 ? "+" : ""}{number(factor.contribution, 3)}
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line" aria-hidden="true">
                      <div
                        className={factor.contribution >= 0 ? "h-full bg-bad" : "h-full bg-good"}
                        style={{ width: `${Math.min(100, Math.abs(factor.contribution) * 400)}%` }}
                      />
                    </div>
                    <p className="mt-1 text-xs text-muted">
                      {directionLabel(factor.direction)}. Value {String(factor.actual ?? "n/a")}
                      {factor.baseline != null ? `, baseline ${String(factor.baseline)}` : ""}.
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">No explanation is stored for this customer. Refresh the prediction to generate one from the loaded model. The page will not invent factor scores.</p>
            )}
        </Panel>
        <div className="space-y-4 lg:col-span-2">
          <Panel title="Recommended next step">
            {prediction?.recommendations?.length ? (
              <ul className="space-y-3">
                {prediction.recommendations.map((item) => (
                  <li key={item.action} className="rounded-xl border border-line p-3">
                    <p className="text-xs uppercase tracking-wide text-muted">{item.effort} effort · {item.category}</p>
                    <p className="mt-1 font-semibold">{item.action}</p>
                    <p className="mt-1 text-sm text-muted">{item.reason}</p>
                    <button type="button" className={`${buttonClass} mt-3`} onClick={() => createFollowUp(item, prediction.priority_score)}>Record follow-up</button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">No recommendation is available until this customer has a score.</p>
            )}
            <p className="mt-3 text-xs leading-5 text-muted">{state.data.decision_support_warning}</p>
          </Panel>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Prediction history" description="Earlier scores stored for this id. A single import usually leaves one current score.">
          {state.data.history.length === 0 && <p className="text-sm text-muted">No prediction history has been recorded.</p>}
          <ul className="space-y-2 text-sm">
            {state.data.history.map((item, index) => (
              <li key={`${item.created_at}-${index}`} className="flex flex-wrap items-center justify-between gap-2 border-b border-line py-2">
                <span>{compactDate(item.created_at)} · {item.source || "score"} · {item.status || "scored"}</span>
                <span className="flex items-center gap-2">{percent(item.probability)} <RiskBadge risk={item.risk_category} /></span>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title="Retention activity">
          {state.data.campaigns.length === 0 && <p className="text-sm text-muted">No follow-up has been recorded. Recording one does not create a measured save rate.</p>}
          <ul className="space-y-3">
            {state.data.campaigns.map((campaign) => (
              <li key={campaign.id} className="rounded-xl border border-line p-3 text-sm">
                <p className="font-semibold">{campaign.action}</p>
                <p className="mt-1 text-muted">{campaign.reason}</p>
                <p className="mt-1 text-xs text-muted">{campaign.campaign_name} · updated {compactDate(campaign.updated_at)} · outcomes unavailable</p>
                <label className="mt-3 block text-xs text-muted">
                  Note
                  <input
                    className={`${controlClass} mt-1`}
                    defaultValue={campaign.notes || ""}
                    onChange={(event) => setNotes((current) => ({ ...current, [campaign.id]: event.target.value }))}
                  />
                </label>
                <label className="mt-2 block text-xs text-muted">
                  Status
                  <select className={`${controlClass} mt-1`} aria-label={`Status for ${campaign.action}`} value={campaign.status} onChange={(event) => updateCampaign(campaign, event.target.value)}>
                    {STATUSES.map((status) => <option key={status} value={status}>{titleCase(status)}</option>)}
                  </select>
                </label>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
