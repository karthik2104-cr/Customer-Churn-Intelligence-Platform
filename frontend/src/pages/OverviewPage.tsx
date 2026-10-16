import { Link } from "react-router-dom";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { RateBars } from "../components/charts";
import { EmptyBlock, ErrorBlock, LoadingBlock, PageHeader, Panel, RiskBadge, Stat } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Summary } from "../types";
import { compactDate, money, number, percent } from "../utils/format";

const RISK_COLORS: Record<string, string> = { High: "#b91c1c", Medium: "#a16207", Low: "#16804a", Unscored: "#7d8b99" };

export function OverviewPage() {
  const state = useApi<Summary>("/api/dashboard/summary");
  if (state.loading) return <LoadingBlock label="Loading overview" />;
  if (state.error) return <ErrorBlock message={state.error} />;
  if (!state.data || state.data.total_customers === 0) {
    return <EmptyBlock title="No customers loaded" body="Import the telco CSV from Data Management, or place it in data/raw and restart the API." />;
  }
  const data = state.data;
  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title="Where is churn showing up?"
        lede="Historical labels and model risk are shown separately. A high-risk count is a current score, not a count of customers who have already left."
      />
      <p className="mb-5 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted">{data.decision_support_warning}</p>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <Stat label="Customers" value={number(data.total_customers)} hint="Rows currently stored from the imported file." />
        <Stat label="Observed churn rate" value={percent(data.observed_churn_rate)} hint="Churned customers divided by customers with a Yes or No label." detail={`${number(data.observed_churn_count)} churned · ${number(data.observed_retained_count)} retained`} />
        <Stat label="High-risk customers" value={number(data.high_risk_count)} hint="Customers whose current predicted probability is at or above the high-risk threshold." detail={data.predictions_available ? "From the loaded model" : "Score the table to populate risk"} />
        <Stat label="Average predicted probability" value={percent(data.average_predicted_probability)} hint="Mean model probability across scored customers. This is not the historical churn rate." />
        <Stat label="Follow-up required" value={number(data.follow_up_required)} hint="Campaign records currently marked follow-up required." />
        <Stat label="Medium risk" value={number(data.medium_risk_count)} hint="Elevated scores below the positive-class threshold." detail={`${number(data.low_risk_count)} low · ${number(data.unscored_count)} unscored`} />
      </div>
      <div className="mt-6 grid gap-4 lg:grid-cols-5">
        <Panel title="Risk breakdown" description="Predicted risk, not historical churn." >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.risk_breakdown} dataKey="customers" nameKey="name" innerRadius={52} outerRadius={80}>
                  {data.risk_breakdown.map((entry) => (
                    <Cell key={entry.name} fill={RISK_COLORS[entry.name] || "#8a8175"} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <div className="lg:col-span-3">
          <Panel title="Churn rate by contract" description={data.denominator_note}>
            <RateBars rows={data.contract_churn} />
          </Panel>
        </div>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <Panel title="Priority customers" description="Ranked by the documented priority heuristic.">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <caption className="sr-only">Highest priority customers</caption>
                <thead className="text-xs uppercase tracking-wide text-muted">
                  <tr>
                    <th className="py-2">Customer</th>
                    <th>Risk</th>
                    <th>Probability</th>
                    <th>Priority</th>
                    <th>Contract</th>
                    <th>Tenure</th>
                    <th>Monthly</th>
                  </tr>
                </thead>
                <tbody>
                  {data.priority_customers.map((row) => (
                    <tr key={row.customer_id} className="border-t border-line">
                      <td className="py-2"><Link className="font-semibold text-accent" to={`/customers/${row.customer_id}`}>{row.customer_id}</Link></td>
                      <td><RiskBadge risk={row.risk_category} /></td>
                      <td>{percent(row.probability)}</td>
                      <td>{number(row.priority_score, 1)}</td>
                      <td>{row.contract}</td>
                      <td>{row.tenure}</td>
                      <td>{money(row.monthly_charges, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
        <div className="lg:col-span-2">
          <Panel title="Recent activity" description={data.time_series_note}>
            <ul className="space-y-3 text-sm">
              {data.recent_activity.length === 0 && <li className="text-muted">No predictions or campaign updates yet.</li>}
              {data.recent_activity.map((item, index) => (
                <li key={`${item.at}-${index}`} className="border-b border-line pb-2">
                  <p>{item.summary}</p>
                  <p className="text-xs text-muted">{compactDate(item.at)}{item.customer_id ? ` · ${item.customer_id}` : ""}</p>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  );
}
