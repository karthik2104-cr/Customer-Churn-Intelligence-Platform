import { Link, useSearchParams } from "react-router-dom";
import { ErrorBlock, Field, LoadingBlock, PageHeader, Panel, RiskBadge, controlClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { CustomerPage } from "../types";
import { money, number, percent } from "../utils/format";

export function CustomersPage() {
  const [params, setParams] = useSearchParams();
  const page = params.get("page") || "1";
  const query = `/api/customers?${params.toString() || "page=1&page_size=25&sort=priority_score&order=desc"}`;
  const state = useApi<CustomerPage>(query);
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "page") next.set("page", "1");
    setParams(next);
  }
  return (
    <div>
      <PageHeader eyebrow="Customer profiles" title="Find a customer and open the story" lede="Sort and filter the stored base. A segment filter uses the same rules as the segment counts, including every condition in that rule. Observed churn is the historical label. Risk and probability come from the latest score." />
      <Panel>
        <div className="grid gap-3 md:grid-cols-4">
          <Field label="Search id"><input className={controlClass} defaultValue={params.get("search") || ""} onBlur={(event) => update("search", event.target.value)} /></Field>
          <Field label="Contract">
            <select className={controlClass} value={params.get("contract") || ""} onChange={(event) => update("contract", event.target.value)}>
              <option value="">All</option>
              <option>Month-to-month</option>
              <option>One year</option>
              <option>Two year</option>
            </select>
          </Field>
          <Field label="Risk">
            <select className={controlClass} value={params.get("risk") || ""} onChange={(event) => update("risk", event.target.value)}>
              <option value="">All</option>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
          </Field>
          <Field label="Internet">
            <select className={controlClass} value={params.get("internet_service") || ""} onChange={(event) => update("internet_service", event.target.value)}>
              <option value="">All</option>
              <option>DSL</option>
              <option>Fiber optic</option>
              <option>No</option>
            </select>
          </Field>
          <Field label="Sort">
            <select className={controlClass} value={params.get("sort") || "priority_score"} onChange={(event) => update("sort", event.target.value)}>
              <option value="priority_score">Priority</option>
              <option value="probability">Probability</option>
              <option value="tenure">Tenure</option>
              <option value="monthly_charges">Monthly charges</option>
              <option value="customer_id">Customer id</option>
            </select>
          </Field>
          <Field label="Order">
            <select className={controlClass} value={params.get("order") || "desc"} onChange={(event) => update("order", event.target.value)}>
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </Field>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          {[...params.entries()].filter(([key, value]) => value && key !== "page" && key !== "page_size").map(([key, value]) => (
            <button key={key} type="button" className="rounded-full border border-line px-2 py-1" onClick={() => update(key, "")}>{key}: {value} · remove</button>
          ))}
          <button type="button" className="rounded-full border border-line px-2 py-1" onClick={() => setParams(new URLSearchParams("page=1&page_size=25&sort=priority_score&order=desc"))}>Reset filters</button>
        </div>
      </Panel>
      <div className="mt-4">
        {state.loading && <LoadingBlock label="Loading customers" />}
        {state.error && <ErrorBlock message={state.error} />}
        {state.data && (
          <Panel title={`${state.data.total.toLocaleString()} customers`} description={state.data.segment ? segmentNote(state.data.total, state.data.segment.members, state.data.segment.name, state.data.segment.definition) : undefined} action={<Pager page={Number(page)} total={state.data.total} pageSize={state.data.page_size} onPage={(value) => update("page", String(value))} />}>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <caption className="sr-only">Customer directory</caption>
                <thead className="text-xs uppercase text-muted">
                  <tr>
                    <th className="py-2">Id</th><th>Risk</th><th>Probability</th><th className="hidden md:table-cell">Priority</th><th className="hidden lg:table-cell">Observed</th><th>Contract</th><th className="hidden sm:table-cell">Tenure</th><th className="hidden md:table-cell">Monthly</th><th>Profile</th>
                  </tr>
                </thead>
                <tbody>
                  {state.data.items.map((row) => (
                    <tr key={row.customer_id} className="border-t border-line">
                      <td className="py-2 font-semibold">{row.customer_id}</td>
                      <td><RiskBadge risk={row.risk_category} /></td>
                      <td>{percent(row.probability)}</td>
                      <td className="hidden md:table-cell">{number(row.priority_score, 1)}</td>
                      <td className="hidden lg:table-cell">{row.observed_churn || "n/a"}</td>
                      <td>{row.contract}</td>
                      <td className="hidden sm:table-cell">{row.tenure}</td>
                      <td className="hidden md:table-cell">{money(row.monthly_charges, 2)}</td>
                      <td><Link className="font-semibold text-accent" to={`/customers/${row.customer_id}`}>Open</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function segmentNote(total: number, members: number, name: string, definition: string): string {
  const match = total === members
    ? `${total.toLocaleString()} customers match every condition.`
    : `${total.toLocaleString()} of ${members.toLocaleString()} segment members remain after the extra directory filters.`;
  return `${name}. ${definition} ${match}`;
}

function Pager({ page, total, pageSize, onPage }: { page: number; total: number; pageSize: number; onPage: (page: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex items-center gap-2 text-sm">
      <button type="button" className="rounded border border-line px-2 py-1" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
      <span>{page} / {pages}</span>
      <button type="button" className="rounded border border-line px-2 py-1" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
    </div>
  );
}
