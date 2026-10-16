import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ChurnRow } from "../types";
import { percent } from "../utils/format";

const tooltipStyle = { background: "rgb(var(--surface))", border: "1px solid rgb(var(--line))", borderRadius: 12, color: "rgb(var(--ink))" };

export function RateBars({ rows }: { rows: ChurnRow[] }) {
  const data = rows.map((row) => ({
    name: row.name,
    rate: row.churn_rate == null ? 0 : Number((row.churn_rate * 100).toFixed(1)),
    customers: row.customers,
    churned: row.churned,
  }));
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16 }}>
          <CartesianGrid stroke="var(--grid)" horizontal={false} />
          <XAxis type="number" unit="%" tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={120} tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value, name) => (name === "rate" ? [`${value}%`, "Churn rate"] : [value, name])}
            labelFormatter={(label) => String(label)}
          />
          <Bar dataKey="rate" name="rate" fill="rgb(var(--accent))" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-muted">Bar length is the churn rate. Counts are in the tooltip and table so a large group is not mistaken for a high rate.</p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Counts behind the churn-rate chart</caption>
          <thead className="text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="py-1">Group</th>
              <th>Customers</th>
              <th>Churned</th>
              <th>Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name} className="border-t border-line">
                <td className="py-1.5">{row.name}</td>
                <td>{row.customers.toLocaleString()}</td>
                <td>{row.churned.toLocaleString()}</td>
                <td>{percent(row.churn_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function CurveChart({ points, xKey, yKey, xLabel, yLabel }: { points: Record<string, number>[]; xKey: string; yKey: string; xLabel: string; yLabel: string }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis dataKey={xKey} tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} label={{ value: xLabel, position: "insideBottom", offset: -2, fill: "rgb(var(--muted))" }} />
          <YAxis tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "rgb(var(--muted))" }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend />
          <Line type="monotone" dataKey={yKey} name={yLabel} stroke="rgb(var(--accent))" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ImportanceBars({ rows }: { rows: { feature: string; importance: number }[] }) {
  const data = rows.map((row) => ({ ...row, importance: Number(row.importance.toFixed(4)) }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid stroke="var(--grid)" horizontal={false} />
          <XAxis type="number" tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} />
          <YAxis type="category" dataKey="feature" width={130} tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey="importance" name="Importance">
            {data.map((row) => (
              <Cell key={row.feature} fill={row.importance >= 0 ? "rgb(var(--bad))" : "rgb(var(--good))"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
