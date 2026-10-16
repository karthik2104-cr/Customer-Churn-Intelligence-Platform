import { Link } from "react-router-dom";
import { ErrorBlock, LoadingBlock, PageHeader, Panel } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { ClusterSegment, SegmentRule } from "../types";
import { money, number, percent } from "../utils/format";

interface Segments {
  high_charge_cutoff: number | null;
  cutoff_note: string;
  rules: SegmentRule[];
  clusters: ClusterSegment[];
  cluster_note: string;
  dataset_label: string | null;
}

export function SegmentsPage() {
  const state = useApi<Segments>("/api/analytics/segments");
  if (state.loading) return <LoadingBlock label="Loading segments" />;
  if (state.error) return <ErrorBlock message={state.error} />;
  if (!state.data) return null;
  const data = state.data;
  return (
    <div>
      <PageHeader eyebrow="Segments" title="Groups you can explain" lede="Rule segments can overlap. Primary counts assign each customer to the first matching rule. Cluster labels describe centers fit without the churn label." />
      <p className="mb-4 text-sm text-muted">{data.cutoff_note} Cutoff {money(data.high_charge_cutoff, 2)}. Dataset: {data.dataset_label || "n/a"}. Opening a rule lists the customers who satisfy every condition of that rule. The list total matches the customer count on the card.</p>
      <div className="grid gap-4 lg:grid-cols-2">
        {data.rules.map((rule) => (
          <Panel key={rule.id} title={rule.name} description={rule.definition}>
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <Metric label="Customers in rule" value={number(rule.customers)} />
              <Metric label="Primary assignment" value={number(rule.primary_customers)} />
              <Metric label="Observed churn rate" value={percent(rule.observed_churn_rate)} />
              <Metric label="Average probability" value={percent(rule.average_predicted_probability)} />
              <Metric label="Average tenure" value={number(rule.average_tenure, 1)} />
              <Metric label="Average monthly" value={money(rule.average_monthly_charges, 2)} />
            </dl>
            <p className="mt-3 text-sm">{rule.suggested_action}</p>
            <Link className="mt-3 inline-flex text-sm font-semibold text-accent" to={`/customers?segment=${encodeURIComponent(rule.id)}&page=1`}>View exact members</Link>
          </Panel>
        ))}
      </div>
      <div className="mt-4">
        <Panel title="Cluster profiles" description={data.cluster_note}>
          {data.clusters.length === 0 && <p className="text-sm text-muted">No cluster artifact is loaded. Rule segments above still work. Retrain to fit clusters.</p>}
          <div className="grid gap-3 md:grid-cols-2">
            {data.clusters.map((cluster) => (
              <article key={cluster.label} className="rounded-xl border border-line p-3 text-sm">
                <h3 className="font-semibold">{cluster.label}</h3>
                <p className="mt-1 text-muted">{number(cluster.customers)} customers · observed churn {percent(cluster.observed_churn_rate)} · average probability {percent(cluster.average_predicted_probability)}</p>
                <p className="text-muted">Tenure {number(cluster.average_tenure, 1)} months · monthly {money(cluster.average_monthly_charges, 2)}</p>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-muted">{label}</dt><dd className="font-semibold">{value}</dd></div>;
}
