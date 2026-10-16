import { FormEvent, useEffect, useState } from "react";
import { ErrorBlock, Field, LoadingBlock, PageHeader, Panel, buttonClass, controlClass, ghostButtonClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, post, put } from "../services/api";
import type { Settings } from "../types";

export function SettingsPage() {
  const state = useApi<Settings>("/api/settings");
  const [medium, setMedium] = useState("0.3");
  const [high, setHigh] = useState("0.5");
  const [margin, setMargin] = useState("0.6");
  const [cost, setCost] = useState("25");
  const [success, setSuccess] = useState("0.2");
  const [horizon, setHorizon] = useState("12");
  const [capacity, setCapacity] = useState("");
  const [revenue, setRevenue] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!state.data) return;
    setMedium(String(state.data.medium_threshold));
    setHigh(String(state.data.high_threshold));
    setMargin(String(state.data.contribution_margin));
    setCost(String(state.data.campaign_cost));
    setSuccess(String(state.data.success_rate));
    setHorizon(String(state.data.horizon_months));
    setCapacity(state.data.intervention_capacity == null ? "" : String(state.data.intervention_capacity));
    setRevenue(state.data.avg_monthly_revenue == null ? "" : state.data.avg_monthly_revenue.toFixed(2));
  }, [state.data]);

  async function save(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setMessage(null);
    if (Number(medium) >= Number(high)) {
      setError("The medium threshold must be below the high threshold.");
      return;
    }
    try {
      await put("/api/settings", {
        medium_threshold: Number(medium),
        high_threshold: Number(high),
        contribution_margin: Number(margin),
        campaign_cost: Number(cost),
        success_rate: Number(success),
        horizon_months: Number(horizon),
        avg_monthly_revenue: revenue ? Number(revenue) : undefined,
        intervention_capacity: capacity ? Number(capacity) : null,
        clear_capacity: capacity === "",
      });
      setMessage("Settings saved. Stored scores were relabeled with the new thresholds.");
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not save settings.");
    }
  }

  async function reloadModel() {
    setError(null);
    try {
      const result = await post<{ model_loaded: boolean; model_error: string | null; selected_model: string | null }>("/api/models/reload", {});
      setMessage(result.model_loaded ? `Reloaded ${result.selected_model || "model"} and rescored stored customers.` : `Model did not load. ${result.model_error || ""}`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Reload failed.");
    }
  }

  if (state.loading) return <LoadingBlock label="Loading settings" />;
  if (state.error) return <ErrorBlock message={state.error} />;
  if (!state.data) return null;
  return (
    <div>
      <PageHeader eyebrow="Settings" title="Thresholds and planning assumptions" lede={state.data.threshold_note} />
      <p className="mb-4 text-sm text-muted">{state.data.decision_support_warning}</p>
      <p className="mb-4 text-sm text-muted">{state.data.excluded_features_reason}</p>
      <form noValidate onSubmit={save}>
        <Panel title="Risk thresholds" description={state.data.thresholds_customized ? "These values have been edited." : "These values still match the training report."}>
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Medium threshold">
              <input className={controlClass} type="number" min={0.01} max={0.99} step="0.01" value={medium} onChange={(event) => setMedium(event.target.value)} />
            </Field>
            <Field label="High / decision threshold">
              <input className={controlClass} type="number" min={0.01} max={0.99} step="0.01" value={high} onChange={(event) => setHigh(event.target.value)} />
            </Field>
          </div>
        </Panel>
        <div className="mt-4">
          <Panel title="Planning assumptions" description="Used by the business impact page. Blank revenue keeps the current stored value.">
            <div className="grid gap-3 md:grid-cols-3">
              <Field label="Average monthly revenue"><input className={controlClass} type="number" min={0} step="0.01" value={revenue} onChange={(event) => setRevenue(event.target.value)} /></Field>
              <Field label="Contribution margin"><input className={controlClass} type="number" min={0} max={1} step="0.01" value={margin} onChange={(event) => setMargin(event.target.value)} /></Field>
              <Field label="Campaign cost"><input className={controlClass} type="number" min={0} step="0.01" value={cost} onChange={(event) => setCost(event.target.value)} /></Field>
              <Field label="Success rate"><input className={controlClass} type="number" min={0} max={1} step="0.01" value={success} onChange={(event) => setSuccess(event.target.value)} /></Field>
              <Field label="Horizon months"><input className={controlClass} type="number" min={1} max={60} value={horizon} onChange={(event) => setHorizon(event.target.value)} /></Field>
              <Field label="Capacity"><input className={controlClass} type="number" min={1} value={capacity} onChange={(event) => setCapacity(event.target.value)} /></Field>
            </div>
          </Panel>
        </div>
        {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
        {message && <p className="mt-3 text-sm text-good">{message}</p>}
        <div className="mt-4 flex flex-wrap gap-3">
          <button className={buttonClass} type="submit">Save settings</button>
          <button className={buttonClass} type="button" onClick={reloadModel}>Reload model and rescore</button>
        </div>
      </form>
      <div className="mt-4">
        <Panel title="Appearance" description="Theme is a display preference stored in this browser. It does not change scores or thresholds.">
          <button
            type="button"
            className={ghostButtonClass}
            onClick={() => {
              const next = !document.documentElement.classList.contains("dark");
              document.documentElement.classList.toggle("dark", next);
              localStorage.setItem("churn-theme", next ? "dark" : "light");
            }}
          >
            Toggle light and dark theme
          </button>
        </Panel>
      </div>
    </div>
  );
}
