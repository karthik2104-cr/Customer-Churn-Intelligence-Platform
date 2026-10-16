import { useState } from "react";
import { ErrorBlock, LoadingBlock, PageHeader, Panel, buttonClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, upload } from "../services/api";
import type { QualityResponse } from "../types";
import { compactDate, number, percent } from "../utils/format";

export function DataPage() {
  const state = useApi<QualityResponse>("/api/data-quality");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setPending(true);
    setError(null);
    setMessage(null);
    try {
      const result = await upload<{ stored_customers: number; dataset_label: string; scored_customers?: number; message?: string }>("/api/data/import", file);
      setMessage(`${result.dataset_label}: stored ${result.stored_customers} customers${result.scored_customers != null ? `, scored ${result.scored_customers}` : ""}. ${result.message || ""}`);
      await state.reload();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Import failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div>
      <PageHeader eyebrow="Data management" title="Inspect the file before trusting a score" lede="Raw rows stay separate from cleaned training data. Outliers are reported and kept. A file named as synthetic, or ids starting with SYN-, is labeled synthetic." />
      <Panel title="Import a CSV" description="Expected columns match the IBM telco schema. Place Telco-Customer-Churn.csv in data/raw if you prefer to load it at startup.">
        <input aria-label="Dataset CSV" type="file" accept=".csv,text/csv" disabled={pending} onChange={(event) => onFile(event.target.files?.[0])} />
        {pending && <p className="mt-2 text-sm text-muted">Importing…</p>}
        {message && <p className="mt-2 text-sm text-good">{message}</p>}
        {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
      </Panel>
      <div className="mt-4">
        {state.loading && <LoadingBlock label="Loading data quality" />}
        {state.error && <ErrorBlock message={state.error} />}
        {state.data && !state.data.available && <p className="text-sm text-muted">{state.data.message}</p>}
        {state.data?.available && state.data.quality && <QualityBody report={state.data} />}
      </div>
    </div>
  );
}

function QualityBody({ report }: { report: QualityResponse }) {
  const quality = report.quality!;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title={report.dataset_label || "Imported file"} description={`${report.filename || "upload"} · ${compactDate(report.imported_at)}`}>
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <Item label="Raw rows" value={number(quality.rows_raw)} />
          <Item label="Cleaned rows" value={number(quality.rows_cleaned)} />
          <Item label="Stored customers" value={number(report.stored_customers)} />
          <Item label="Duplicate ids" value={number(quality.duplicate_customer_ids)} />
          <Item label="Observed churn rate" value={percent(quality.observed_churn_rate)} />
          <Item label="Class counts" value={Object.entries(quality.class_counts).map(([key, value]) => `${key} ${value}`).join(" · ") || "n/a"} />
        </dl>
        <p className="mt-3 text-sm text-muted">{quality.outlier_policy}</p>
      </Panel>
      <Panel title="Missing and invalid values">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-muted"><tr><th className="py-2">Column</th><th>Missing</th><th>Invalid numeric</th></tr></thead>
          <tbody>
            {Object.entries(quality.missing_values).map(([column, missing]) => (
              <tr key={column} className="border-t border-line">
                <td className="py-1.5">{column}</td>
                <td>{missing}</td>
                <td>{quality.invalid_numeric_values[column] ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <Panel title="Outlier inspection" description="Flagged rows are still in the customer table.">
        <ul className="space-y-2 text-sm">
          {Object.entries(quality.outliers).map(([column, detail]) => (
            <li key={column}><span className="font-semibold">{column}</span>: {detail.flagged} flagged. {detail.note}</li>
          ))}
        </ul>
      </Panel>
      <Panel title="Cleaning notes">
        <ul className="list-disc space-y-2 pl-5 text-sm text-muted">
          {quality.cleaning_notes.length === 0 && <li>No extra cleaning notes.</li>}
          {quality.cleaning_notes.map((note) => <li key={note}>{note}</li>)}
        </ul>
      </Panel>
    </div>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-muted">{label}</dt><dd className="font-semibold">{value}</dd></div>;
}
