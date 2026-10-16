import { FormEvent, useEffect, useState } from "react";
import { ErrorBlock, Field, Hint, LoadingBlock, PageHeader, Panel, buttonClass, controlClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, put } from "../services/api";
import type { Impact, Scenario, Settings } from "../types";
import { money, number, percent } from "../utils/format";

export function ImpactPage() {
  const impact = useApi<Impact>("/api/business-impact");
  const settings = useApi<Settings>("/api/settings");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [form, setForm] = useState({
    avg_monthly_revenue: "",
    contribution_margin: "0.6",
    campaign_cost: "25",
    success_rate: "0.2",
    horizon_months: "12",
    intervention_capacity: "",
  });

  useEffect(() => {
    if (!settings.data) return;
    setForm({
      avg_monthly_revenue: settings.data.avg_monthly_revenue == null ? "" : settings.data.avg_monthly_revenue.toFixed(2),
      contribution_margin: String(settings.data.contribution_margin),
      campaign_cost: String(settings.data.campaign_cost),
      success_rate: String(settings.data.success_rate),
      horizon_months: String(settings.data.horizon_months),
      intervention_capacity: settings.data.intervention_capacity == null ? "" : String(settings.data.intervention_capacity),
    });
  }, [settings.data]);

  async function save(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSaved(false);
    const revenue = Number(form.avg_monthly_revenue);
    if (!form.avg_monthly_revenue || revenue <= 0) {
      setError("Average monthly revenue must be greater than zero.");
      return;
    }
    try {
      await put("/api/settings", {
        avg_monthly_revenue: revenue,
        contribution_margin: Number(form.contribution_margin),
        campaign_cost: Number(form.campaign_cost),
        success_rate: Number(form.success_rate),
        horizon_months: Number(form.horizon_months),
        intervention_capacity: form.intervention_capacity ? Number(form.intervention_capacity) : null,
        clear_capacity: form.intervention_capacity === "",
      });
      setSaved(true);
      await impact.reload();
      await settings.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not save assumptions.");
    }
  }

  if (impact.loading || settings.loading) return <LoadingBlock label="Loading impact scenarios" />;
  if (impact.error) return <ErrorBlock message={impact.error} />;
  if (!impact.data) return null;
  const data = impact.data;
  return (
    <div>
      <PageHeader eyebrow="Business impact" title="Estimate exposure under explicit assumptions" lede="These figures are scenarios. They are not observed savings, and a predicted churner is not treated as a certain loss." />
      <p className="mb-4 text-sm text-muted">{data.disclaimer || data.assumptions_note}</p>
      <p className="mb-4 text-sm text-muted">{data.outcomes_note}</p>
      <form noValidate onSubmit={save}>
        <Panel title="Assumptions" description="Editing these values recalculates every scenario. The dataset does not contain cost or save-rate fields.">
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="Average monthly revenue">
              <input className={controlClass} type="number" min={0} step="0.01" value={form.avg_monthly_revenue} onChange={(event) => setForm({ ...form, avg_monthly_revenue: event.target.value })} />
            </Field>
            <Field label="Contribution margin (0–1)">
              <input className={controlClass} type="number" min={0} max={1} step="0.01" value={form.contribution_margin} onChange={(event) => setForm({ ...form, contribution_margin: event.target.value })} />
            </Field>
            <Field label="Campaign cost per customer">
              <input className={controlClass} type="number" min={0} step="0.01" value={form.campaign_cost} onChange={(event) => setForm({ ...form, campaign_cost: event.target.value })} />
            </Field>
            <Field label="Base success rate (0–1)">
              <input className={controlClass} type="number" min={0} max={1} step="0.01" value={form.success_rate} onChange={(event) => setForm({ ...form, success_rate: event.target.value })} />
            </Field>
            <Field label="Horizon (months)">
              <input className={controlClass} type="number" min={1} max={60} value={form.horizon_months} onChange={(event) => setForm({ ...form, horizon_months: event.target.value })} />
            </Field>
            <Field label="Intervention capacity (optional)">
              <input className={controlClass} type="number" min={1} value={form.intervention_capacity} onChange={(event) => setForm({ ...form, intervention_capacity: event.target.value })} />
            </Field>
          </div>
          {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
          <button className={`${buttonClass} mt-4`} type="submit">Recalculate</button>
          {saved && <p className="mt-2 text-sm text-good">Assumptions saved. The table uses the new values.</p>}
        </Panel>
      </form>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <Mini label="Customers in table" value={number(data.reference.customers_total)} hint="Stored customers, scored or not." />
        <Mini label="At risk before capacity" value={number(data.reference.customers_at_risk_before_capacity)} hint="Medium and high risk customers before the capacity limit." />
        <Mini label="Data average monthly charges" value={money(data.reference.data_average_monthly_charges, 2)} hint="Mean MonthlyCharges in the imported table. Use it as a reference, not as realized revenue." />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        {data.scenarios.map((scenario) => <ScenarioCard key={scenario.name} scenario={scenario} />)}
      </div>
      <div className="mt-4">
        <Panel title="Formulas">
          <ul className="list-disc space-y-2 pl-5 text-sm text-muted">
            {data.formulas.map((formula) => <li key={formula}>{formula}</li>)}
          </ul>
        </Panel>
      </div>
    </div>
  );
}

function Mini({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-2xl border border-line bg-surface p-4">
      <div className="flex items-center justify-between"><p className="text-sm text-muted">{label}</p><Hint text={hint} /></div>
      <p className="mt-1 font-serif text-2xl">{value}</p>
    </div>
  );
}

function ScenarioCard({ scenario }: { scenario: Scenario }) {
  return (
    <Panel title={scenario.name}>
      <dl className="space-y-2 text-sm">
        <Row label="Success rate used" value={percent(scenario.success_rate)} />
        <Row label="Customers targeted" value={number(scenario.customers_targeted)} />
        <Row label="Expected churners" value={number(scenario.expected_churners, 1)} />
        <Row label="Revenue exposure" value={money(scenario.revenue_exposure, 0)} />
        <Row label="Contribution exposure" value={money(scenario.contribution_exposure, 0)} />
        <Row label="Expected retained" value={number(scenario.expected_retained_customers, 1)} />
        <Row label="Contribution benefit" value={money(scenario.estimated_contribution_benefit, 0)} />
        <Row label="Campaign cost" value={money(scenario.campaign_cost, 0)} />
        <Row label="Estimated net benefit" value={money(scenario.estimated_net_benefit, 0)} />
        <Row label="Break-even success" value={percent(scenario.break_even_success_rate)} />
      </dl>
      {scenario.success_rate_capped_at_one && <p className="mt-2 text-xs text-muted">The scenario multiplier was capped at a 100% success rate.</p>}
    </Panel>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-3"><dt className="text-muted">{label}</dt><dd className="font-semibold">{value}</dd></div>;
}
