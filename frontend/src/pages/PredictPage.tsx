import { FormEvent, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ErrorBlock, Field, PageHeader, Panel, RiskBadge, buttonClass, controlClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, post } from "../services/api";
import type { ModelInfo, Prediction } from "../types";
import { percent } from "../utils/format";

const DEFAULTS: Record<string, string[]> = {
  Partner: ["Yes", "No"],
  Dependents: ["Yes", "No"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["No", "Yes", "No phone service"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["No", "Yes", "No internet service"],
  OnlineBackup: ["No", "Yes", "No internet service"],
  DeviceProtection: ["No", "Yes", "No internet service"],
  TechSupport: ["No", "Yes", "No internet service"],
  StreamingTV: ["No", "Yes", "No internet service"],
  StreamingMovies: ["No", "Yes", "No internet service"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
};

const INITIAL: Record<string, string> = {
  customer_id: "",
  tenure: "12",
  MonthlyCharges: "70",
  TotalCharges: "840",
  Partner: "No",
  Dependents: "No",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
};

export function PredictPage() {
  const models = useApi<ModelInfo>("/api/models");
  const options = models.data?.feature_spec?.categorical || DEFAULTS;
  const [form, setForm] = useState(INITIAL);
  const [result, setResult] = useState<Prediction | null>(null);
  const [previous, setPrevious] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const fields = useMemo(() => Object.keys(options), [options]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (!form.tenure || Number(form.tenure) < 0) {
      setError("Tenure must be zero or greater.");
      return;
    }
    setPending(true);
    try {
      const payload = {
        ...form,
        customer_id: form.customer_id || null,
        tenure: Number(form.tenure),
        MonthlyCharges: Number(form.MonthlyCharges),
        TotalCharges: form.TotalCharges === "" ? null : Number(form.TotalCharges),
      };
      const next = await post<Prediction>("/api/predictions/single", payload);
      setPrevious(result?.probability ?? null);
      setResult(next);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Prediction failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div>
      <PageHeader eyebrow="Prediction" title="Score one customer" lede="The result is a predicted probability and a risk band. Changing a field shows how the score moves. That comparison is not a causal estimate of an intervention." />
      {!models.data?.model_loaded && <p className="mb-4 rounded-xl border border-warn/40 bg-warn/10 p-3 text-sm">The model is not loaded. Train it, then use Settings to reload. {models.data?.model_error}</p>}
      <div className="grid gap-4 lg:grid-cols-5">
        <form className="lg:col-span-3" noValidate onSubmit={onSubmit}>
          <Panel title="Customer details">
            <div className="grid gap-3 md:grid-cols-2">
              <Field label="Customer id (optional)">
                <input className={controlClass} value={form.customer_id} onChange={(event) => setForm({ ...form, customer_id: event.target.value })} />
              </Field>
              <Field label="Tenure (months)">
                <input className={controlClass} required type="number" min={0} value={form.tenure} onChange={(event) => setForm({ ...form, tenure: event.target.value })} />
              </Field>
              <Field label="Monthly charges">
                <input className={controlClass} required type="number" min={0} step="0.01" value={form.MonthlyCharges} onChange={(event) => setForm({ ...form, MonthlyCharges: event.target.value })} />
              </Field>
              <Field label="Total charges">
                <input className={controlClass} type="number" min={0} step="0.01" value={form.TotalCharges} onChange={(event) => setForm({ ...form, TotalCharges: event.target.value })} />
              </Field>
              {fields.map((field) => (
                <Field key={field} label={field}>
                  <select className={controlClass} value={form[field] || options[field][0]} onChange={(event) => setForm({ ...form, [field]: event.target.value })}>
                    {options[field].map((option) => <option key={option}>{option}</option>)}
                  </select>
                </Field>
              ))}
            </div>
            {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
            <button className={`${buttonClass} mt-4`} type="submit" disabled={pending}>{pending ? "Scoring…" : "Score customer"}</button>
          </Panel>
        </form>
        <div className="lg:col-span-2">
          <Panel title="Result">
            {!result && <p className="text-sm text-muted">Submit the form to calculate a probability.</p>}
            {result && (
              <div className="space-y-3">
                <p className="font-serif text-5xl">{percent(result.probability)}</p>
                <RiskBadge risk={result.risk_category} />
                <p className="text-sm">Predicted class: {result.predicted_class}. Priority {result.priority_score ?? "n/a"}.</p>
                {previous != null && (
                  <p className="text-sm text-muted">Previous score {percent(previous)}. The difference shows how this model responds to the edited inputs. It does not estimate the effect of changing the customer's contract or services in the real world.</p>
                )}
                <p className="text-sm text-muted">{result.reliability.message}</p>
                <p className="text-sm text-muted">Method: {result.explanation_method}. {result.explanation_fallback}</p>
                <div>
                  <p className="text-sm font-semibold">Primary action</p>
                  <p className="text-sm">{result.recommendations[0]?.action}</p>
                  <p className="mt-1 text-sm text-muted">{result.recommendations[0]?.reason}</p>
                </div>
                {result.customer_id && <Link className="inline-flex text-sm font-semibold text-accent" to={`/customers/${result.customer_id}`}>Open customer profile</Link>}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
