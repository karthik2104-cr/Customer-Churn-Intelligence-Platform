import type { ReactNode } from "react";
import { Info } from "lucide-react";

export function Panel({ title, description, children, action, className = "" }: { title?: string; description?: string; children: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-line bg-surface p-5 shadow-card ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="font-serif text-xl text-ink">{title}</h2>}
            {description && <p className="mt-1 text-sm text-muted">{description}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, detail }: { label: string; value: string; hint: string; detail?: string }) {
  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-card">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-muted">{label}</p>
        <Hint text={hint} />
      </div>
      <p className="mt-2 font-serif text-3xl tracking-tight text-ink">{value}</p>
      {detail && <p className="mt-1 text-xs text-muted">{detail}</p>}
    </div>
  );
}

export function Hint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <button type="button" className="text-muted" aria-label={text}>
        <Info size={15} />
      </button>
      <span role="tooltip" className="pointer-events-none absolute right-0 top-6 z-20 hidden w-64 rounded-lg border border-line bg-surface p-2 text-left text-xs leading-5 text-ink shadow-card group-focus-within:block group-hover:block">
        {text}
      </span>
    </span>
  );
}

export function RiskBadge({ risk }: { risk: string | null | undefined }) {
  const tone =
    risk === "High"
      ? "bg-bad/10 text-bad"
      : risk === "Medium"
        ? "bg-warn/10 text-warn"
        : risk === "Low"
          ? "bg-good/10 text-good"
          : "bg-line text-muted";
  const mark = risk === "High" ? "H" : risk === "Medium" ? "M" : risk === "Low" ? "L" : "–";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${tone}`}>
      <span aria-hidden="true" className="inline-grid h-4 w-4 place-items-center rounded-sm border border-current text-[10px] leading-none">
        {mark}
      </span>
      {risk || "Unscored"}
    </span>
  );
}

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-line bg-surface p-8 text-sm text-muted" role="status">
      {label}…
    </div>
  );
}

export function ErrorBlock({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-bad/30 bg-bad/10 p-4 text-sm text-bad" role="alert">
      {message}
    </div>
  );
}

export function EmptyBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-line bg-surface p-8">
      <h2 className="font-serif text-xl">{title}</h2>
      <p className="mt-2 max-w-xl text-sm text-muted">{body}</p>
    </div>
  );
}

export function PageHeader({ eyebrow, title, lede }: { eyebrow: string; title: string; lede: string }) {
  return (
    <header className="mb-6">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-accent">{eyebrow}</p>
      <h1 className="mt-1 font-serif text-3xl text-ink md:text-4xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">{lede}</p>
    </header>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-muted">{label}</span>
      {children}
    </label>
  );
}

export const controlClass = "w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink";
export const buttonClass = "inline-flex min-h-10 items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-accent-ink disabled:opacity-60";
export const ghostButtonClass = "inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-semibold text-ink";
