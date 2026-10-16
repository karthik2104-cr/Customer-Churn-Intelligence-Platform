import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ErrorBlock, Field, PageHeader, Panel, RiskBadge, buttonClass, controlClass, ghostButtonClass } from "../components/ui";
import { ApiError, downloadCsv, upload } from "../services/api";
import type { Prediction } from "../types";
import { percent } from "../utils/format";

interface BatchResponse {
  batch_id: string;
  rows: number;
  scored: number;
  invalid: number;
  notes: string[];
  preview: Prediction[];
  decision_support_warning: string;
}

export function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [rows, setRows] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [risk, setRisk] = useState("");
  const [sort, setSort] = useState("probability");
  const visible = useMemo(() => {
    const filtered = rows.filter((row) => !risk || row.risk_category === risk);
    return [...filtered].sort((left, right) => {
      const a = Number(left[sort as "probability"] ?? -1);
      const b = Number(right[sort as "probability"] ?? -1);
      return b - a;
    });
  }, [rows, risk, sort]);

  async function submit() {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setPending(true);
    setError(null);
    try {
      const response = await upload<BatchResponse>("/api/predictions/batch", file);
      setResult(response);
      const page = await fetch(`/api/predictions/batches/${response.batch_id}?page_size=100`).then((res) => res.json());
      setRows(page.items || response.preview);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Upload failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div>
      <PageHeader eyebrow="Batch prediction" title="Score a file of customers" lede="Invalid rows stay in the result with the reason they were not scored. They are not dropped quietly. The template lists the expected columns and does not contain customer records." />
      <Panel title="Upload">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="CSV file">
            <input className={controlClass} type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          </Field>
          <button type="button" className={buttonClass} disabled={pending} onClick={submit}>{pending ? "Scoring…" : "Score file"}</button>
          <button type="button" className={ghostButtonClass} onClick={downloadTemplate}>Download column template</button>
          {result && <button type="button" className={ghostButtonClass} onClick={() => downloadCsv(`/api/predictions/batches/${result.batch_id}/download`, "churn-batch.csv")}>Download results</button>}
        </div>
        {file && <p className="mt-2 text-sm text-muted">{file.name} · {(file.size / 1024).toFixed(1)} KB</p>}
        {error && <div className="mt-3"><ErrorBlock message={error} /></div>}
      </Panel>
      {result && (
        <div className="mt-4 space-y-4">
          <p className="text-sm text-muted">{result.decision_support_warning}</p>
          <div className="grid gap-3 md:grid-cols-3">
            <Summary label="Rows" value={String(result.rows)} />
            <Summary label="Scored" value={String(result.scored)} />
            <Summary label="Invalid" value={String(result.invalid)} />
          </div>
          <Panel title="Results" action={
            <div className="flex gap-2">
              <select className={controlClass} value={risk} onChange={(event) => setRisk(event.target.value)} aria-label="Filter by risk">
                <option value="">All risks</option>
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
              <select className={controlClass} value={sort} onChange={(event) => setSort(event.target.value)} aria-label="Sort results">
                <option value="probability">Probability</option>
                <option value="priority_score">Priority</option>
              </select>
            </div>
          }>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <caption className="sr-only">Batch scores</caption>
                <thead className="text-xs uppercase text-muted"><tr><th className="py-2">Row</th><th>Customer</th><th>Status</th><th>Risk</th><th>Probability</th><th>Errors</th></tr></thead>
                <tbody>
                  {visible.map((row) => (
                    <tr key={`${row.row_number}-${row.customer_id}`} className="border-t border-line">
                      <td className="py-2">{row.row_number}</td>
                      <td>{row.status === "scored" && row.customer_id ? <Link className="font-semibold text-accent" to={`/customers/${row.customer_id}`}>{row.customer_id}</Link> : (row.customer_id || "n/a")}</td>
                      <td>{row.status}</td>
                      <td><RiskBadge risk={row.risk_category} /></td>
                      <td>{row.probability == null ? "n/a" : percent(row.probability)}</td>
                      <td>{row.errors?.join(" ") || row.factors?.[0]?.feature || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

function downloadTemplate() {
  const headers = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"];
  const blob = new Blob([`${headers.join(",")}\n`], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "churn-batch-template.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div className="rounded-2xl border border-line bg-surface p-4"><p className="text-xs uppercase text-muted">{label}</p><p className="font-serif text-3xl">{value}</p></div>;
}
