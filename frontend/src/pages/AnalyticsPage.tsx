import { useMemo, useState } from "react";
import { RateBars } from "../components/charts";
import { ErrorBlock, Field, LoadingBlock, PageHeader, Panel, controlClass, ghostButtonClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Analytics, ChurnRow } from "../types";
import { number, percent } from "../utils/format";

const FILTERS = [
  ["contract", "Contract", ["", "Month-to-month", "One year", "Two year"]],
  ["internet_service", "Internet", ["", "DSL", "Fiber optic", "No"]],
  ["payment_method", "Payment", ["", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]],
  ["paperless_billing", "Paperless", ["", "Yes", "No"]],
  ["tech_support", "Tech support", ["", "Yes", "No", "No internet service"]],
  ["senior_citizen", "Senior citizen", ["", "0", "1"]],
] as const;

export function AnalyticsPage() {
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [tenureMin, setTenureMin] = useState("");
  const [tenureMax, setTenureMax] = useState("");
  const query = useMemo(() => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => value && params.set(key, value));
    if (tenureMin) params.set("tenure_min", tenureMin);
    if (tenureMax) params.set("tenure_max", tenureMax);
    const text = params.toString();
    return `/api/analytics/churn${text ? `?${text}` : ""}`;
  }, [filters, tenureMin, tenureMax]);
  const state = useApi<Analytics>(query);
  return (
    <div>
      <PageHeader eyebrow="Customer analytics" title="How churn differs across the base" lede="Every rate uses the labeled customers inside the current filter as its denominator. Gender and senior-citizen views are descriptive and are not used for scoring." />
      <Panel title="Filters">
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-4">
          {FILTERS.map(([key, label, options]) => (
            <Field key={key} label={label}>
              <select className={controlClass} value={filters[key] || ""} onChange={(event) => setFilters((current) => ({ ...current, [key]: event.target.value }))}>
                {options.map((option) => <option key={option || "all"} value={option}>{option || "All"}</option>)}
              </select>
            </Field>
          ))}
          <Field label="Tenure min">
            <input className={controlClass} type="number" min={0} value={tenureMin} onChange={(event) => setTenureMin(event.target.value)} />
          </Field>
          <Field label="Tenure max">
            <input className={controlClass} type="number" min={0} value={tenureMax} onChange={(event) => setTenureMax(event.target.value)} />
          </Field>
        </div>
        <button
          type="button"
          className={`${ghostButtonClass} mt-3`}
          onClick={() => {
            setFilters({});
            setTenureMin("");
            setTenureMax("");
          }}
        >
          Reset filters
        </button>
      </Panel>
      <div className="mt-4">
        {state.loading && <LoadingBlock label="Updating charts" />}
        {state.error && <ErrorBlock message={state.error} />}
        {state.data && <AnalyticsBody data={state.data} />}
      </div>
    </div>
  );
}

function AnalyticsBody({ data }: { data: Analytics }) {
  const sections: [string, ChurnRow[]][] = [
    ["Contract", data.by_contract],
    ["Tenure", data.by_tenure],
    ["Monthly charges", data.by_monthly_charges],
    ["Total charges", data.by_total_charges],
    ["Internet service", data.by_internet_service],
    ["Payment method", data.by_payment_method],
    ["Paperless billing", data.by_paperless_billing],
    ["Tech support", data.by_tech_support],
    ["Online security", data.by_online_security],
    ["Add-on services", data.by_addon_group],
    ["Senior citizen (descriptive)", data.by_senior_citizen],
    ["Gender (descriptive)", data.by_gender],
    ["Partner", data.by_partner],
    ["Dependents", data.by_dependents],
  ];
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-4">
        <Metric label="Population in filter" value={number(data.population)} />
        <Metric label="Labeled customers" value={number(data.labeled_customers)} />
        <Metric label="Churned" value={number(data.churned)} />
        <Metric label="Churn rate" value={percent(data.churn_rate)} />
      </div>
      <p className="text-sm text-muted">{data.denominator_note} {data.protected_attribute_note}</p>
      <div className="grid gap-4 xl:grid-cols-2">
        {sections.map(([title, rows]) => (
          <Panel key={title} title={title}>
            {rows.length ? <RateBars rows={rows} /> : <p className="text-sm text-muted">No labeled rows in this filter.</p>}
          </Panel>
        ))}
      </div>
      <Panel title="Numeric profiles by observed churn" description="Means and medians among labeled customers. These are descriptions, not model effects.">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-muted"><tr><th className="py-2">Feature</th><th>Retained mean</th><th>Churned mean</th><th>Retained median</th><th>Churned median</th></tr></thead>
            <tbody>
              {Object.entries(data.numeric_by_churn).map(([feature, groups]) => (
                <tr key={feature} className="border-t border-line">
                  <td className="py-2">{feature}</td>
                  <td>{number(groups.No?.mean, 2)}</td>
                  <td>{number(groups.Yes?.mean, 2)}</td>
                  <td>{number(groups.No?.median, 2)}</td>
                  <td>{number(groups.Yes?.median, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel title="Correlation" description="Pearson correlation on the filtered labeled rows, including the churn flag. Association only.">
        <ul className="grid gap-2 text-sm md:grid-cols-2">
          {data.correlation.map((pair) => (
            <li key={`${pair.left}-${pair.right}`} className="flex justify-between border-b border-line py-1">
              <span>{pair.left} and {pair.right}</span>
              <span>{number(pair.correlation, 3)}</span>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-2xl border border-line bg-surface p-4"><p className="text-xs uppercase tracking-wide text-muted">{label}</p><p className="mt-1 font-serif text-2xl">{value}</p></div>;
}
