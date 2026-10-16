import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { ImportanceBars } from "../components/charts";
import { ErrorBlock, Field, LoadingBlock, PageHeader, Panel, RiskBadge, buttonClass, controlClass } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { ApiError, get } from "../services/api";
import type { Factor, GlobalExplanation, Prediction } from "../types";
import { number, percent } from "../utils/format";

export function ExplainPage() {
  const state = useApi<GlobalExplanation>("/api/explanations/global");
  const [customerId, setCustomerId] = useState("");
  const [local, setLocal] = useState<Prediction | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function lookup(event: FormEvent) {
    event.preventDefault();
    if (!customerId.trim()) {
      setLocalError("Enter a stored customer id.");
      return;
    }
    setPending(true);
    setLocalError(null);
    try {
      setLocal(await get<Prediction>(`/api/explanations/${encodeURIComponent(customerId.trim())}`));
    } catch (caught) {
      setLocal(null);
      setLocalError(caught instanceof ApiError ? caught.message : "Explanation failed.");
    } finally {
      setPending(false);
    }
  }

  if (state.loading) return <LoadingBlock label="Loading explanations" />;
  if (state.error) return <ErrorBlock message={state.error} />;
  if (!state.data) return null;
  const importance = state.data.feature_importance?.features || [];
  return (
    <div>
      <PageHeader eyebrow="Explainability" title="Which inputs the model uses" lede="Importance ranks association with the model's ranking. It does not show that a feature causes a customer to leave." />
      <p className="mb-4 text-sm text-muted">{state.data.limitation}</p>
      <p className="mb-4 text-sm text-muted">{state.data.note} {state.data.exclusion_reason}</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Global permutation importance" description={state.data.feature_importance?.note}>
          {importance.length ? <ImportanceBars rows={importance} /> : <p className="text-sm text-muted">Train a model to produce importance values.</p>}
        </Panel>
        <Panel title="Training-split associations" description={state.data.associations?.caveat}>
          {!state.data.associations && <p className="text-sm text-muted">No association table was saved with the model.</p>}
          {state.data.associations && (
            <div className="space-y-3 text-sm">
              <p>Training churn rate {percent(state.data.associations.training_churn_rate)}.</p>
              <ul>
                {state.data.associations.numeric.map((row) => (
                  <li key={row.feature}>{row.feature}: correlation {number(row.correlation_with_churn, 3)}</li>
                ))}
              </ul>
              {state.data.associations.categorical.slice(0, 4).map((group) => (
                <div key={group.feature}>
                  <p className="font-semibold">{group.feature}</p>
                  <p className="text-muted">{group.levels.slice(0, 3).map((level) => `${level.level} ${percent(level.churn_rate)} (${number(level.customers)})`).join(" · ")}</p>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
      <div className="mt-4">
        <Panel title="One customer's factors">
          <form className="flex flex-wrap items-end gap-3" onSubmit={lookup}>
            <Field label="Customer id">
              <input className={controlClass} value={customerId} onChange={(event) => setCustomerId(event.target.value)} />
            </Field>
            <button className={buttonClass} type="submit" disabled={pending}>{pending ? "Explaining…" : "Explain"}</button>
          </form>
          {localError && <div className="mt-3"><ErrorBlock message={localError} /></div>}
          {local && (
            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <p className="font-serif text-4xl">{percent(local.probability)}</p>
                <RiskBadge risk={local.risk_category} />
                <Link className="text-sm text-accent" to={`/customers/${customerId.trim()}`}>Open profile</Link>
              </div>
              <p className="text-sm text-muted">Method: {local.explanation_method}. {local.explanation_fallback}</p>
              <p className="text-sm text-muted">{local.reliability?.message}</p>
              {local.factors?.length ? <ImportanceBars rows={local.factors.map((factor: Factor) => ({ feature: factor.feature, importance: factor.contribution }))} /> : null}
              <ul className="space-y-1 text-sm">
                {local.factors?.map((factor) => (
                  <li key={factor.feature}>{factor.feature}: actual {String(factor.actual)} versus baseline {String(factor.baseline)} · {factor.direction.replaceAll("_", " ")}</li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
