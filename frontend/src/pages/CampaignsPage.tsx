import { useState } from "react";
import { Link } from "react-router-dom";
import { ErrorBlock, Field, LoadingBlock, PageHeader, Panel, buttonClass, controlClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, patch, post } from "../services/api";
import type { Campaign } from "../types";
import { titleCase } from "../utils/format";

const STATUSES = ["follow_up_required", "contacted", "offer_proposed", "retained"];

export function CampaignsPage() {
  const [status, setStatus] = useState("");
  const [limit, setLimit] = useState(25);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const state = useApi<{ items: Campaign[]; outcomes_available: boolean }>(`/api/campaigns${status ? `?status=${status}` : ""}`);

  async function suggest() {
    setError(null);
    try {
      const result = await post<{ created: number }>("/api/campaigns/suggest", { limit, campaign_name: "Retention queue" });
      setMessage(`Created ${result.created} follow-ups. Outcomes are still unavailable.`);
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not build the queue.");
    }
  }

  async function changeStatus(id: number, next: string) {
    await patch(`/api/campaigns/${id}`, { status: next });
    await state.reload();
  }

  return (
    <div>
      <PageHeader eyebrow="Retention" title="Work a queue without inventing results" lede="Marking a customer retained records your follow-up. It is not a measured save, because this dataset has no campaign outcome field." />
      <Panel title="Suggested queue">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="How many">
            <input className={controlClass} type="number" min={1} max={200} value={limit} onChange={(event) => setLimit(Number(event.target.value))} />
          </Field>
          <button type="button" className={buttonClass} onClick={suggest}>Create follow-ups for elevated risk</button>
        </div>
        {message && <p className="mt-3 text-sm text-good">{message}</p>}
        {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
      </Panel>
      <div className="mt-4">
        <Panel title="Campaigns" action={
          <select className={controlClass} value={status} aria-label="Filter by status" onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {STATUSES.map((item) => <option key={item} value={item}>{titleCase(item)}</option>)}
          </select>
        }>
          {state.loading && <LoadingBlock label="Loading campaigns" />}
          {state.error && <ErrorBlock message={state.error} />}
          {state.data && state.data.items.length === 0 && <p className="text-sm text-muted">No campaign records for this filter.</p>}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <caption className="sr-only">Retention campaigns</caption>
              <thead className="text-xs uppercase text-muted"><tr><th className="py-2">Customer</th><th>Action</th><th>Effort</th><th>Priority</th><th>Status</th></tr></thead>
              <tbody>
                {state.data?.items.map((item) => (
                  <tr key={item.id} className="border-t border-line">
                    <td className="py-2"><Link className="text-accent" to={`/customers/${item.customer_id}`}>{item.customer_id}</Link></td>
                    <td><p className="font-semibold">{item.action}</p><p className="max-w-xl text-muted">{item.reason}</p></td>
                    <td>{item.effort}</td>
                    <td>{item.priority_score.toFixed(1)}</td>
                    <td>
                      <select className={controlClass} aria-label={`Status for ${item.customer_id}`} value={item.status} onChange={(event) => changeStatus(item.id, event.target.value)}>
                        {STATUSES.map((option) => <option key={option} value={option}>{titleCase(option)}</option>)}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}
