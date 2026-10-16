import { CurveChart, ImportanceBars } from "../components/charts";
import { EmptyBlock, ErrorBlock, LoadingBlock, PageHeader, Panel } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Performance } from "../types";
import { number, percent, titleCase } from "../utils/format";

export function ModelPage() {
  const state = useApi<Performance>("/api/models/performance");
  if (state.loading) return <LoadingBlock label="Loading model report" />;
  if (state.error) return <EmptyBlock title="Model report unavailable" body={state.error + " Train with python -m ml.training.train and reload the API."} />;
  if (!state.data) return null;
  const data = state.data;
  const test = data.test_metrics;
  const roc = (test.curves?.roc_curve.fpr || []).map((value, index) => ({ fpr: value, tpr: test.curves?.roc_curve.tpr[index] || 0 }));
  const pr = (test.curves?.pr_curve.recall || []).map((value, index) => ({ recall: value, precision: test.curves?.pr_curve.precision[index] || 0 }));
  const calibration = (test.curves?.calibration_curve.mean_predicted || []).map((value, index) => ({
    mean_predicted: value,
    fraction_positive: test.curves?.calibration_curve.fraction_positive[index] || 0,
  }));
  const matrix = test.confusion_matrix;
  return (
    <div>
      <PageHeader eyebrow="Model performance" title={titleCase(data.selected_model)} lede={data.selection_reason} />
      <div className="grid gap-4 md:grid-cols-4">
        <Metric label="Held-out average precision" value={number(test.average_precision, 3)} />
        <Metric label="Held-out ROC-AUC" value={number(test.roc_auc, 3)} />
        <Metric label="Recall at decision threshold" value={percent(test.recall)} />
        <Metric label="F1 at decision threshold" value={number(test.f1, 3)} />
      </div>
      <p className="mt-3 text-sm text-muted">Comparison rows are cross-validation scores on the training split. Only the selected model has held-out test metrics. Accuracy, precision, recall, and F1 in the comparison table use a 0.5 cutoff. The business threshold is separate.</p>
      <div className="mt-4 overflow-x-auto">
        <Panel title="Training-split comparison">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-muted">
              <tr><th className="py-2">Model</th><th>CV AP</th><th>CV ROC-AUC</th><th>Recall @ 0.5</th><th>Precision @ 0.5</th><th>F1 @ 0.5</th><th>Accuracy @ 0.5</th><th>Seconds</th></tr>
            </thead>
            <tbody>
              {data.comparison.filter((row) => row.status === "ok").map((row) => (
                <tr key={row.name} className="border-t border-line">
                  <td className="py-2 font-semibold">{titleCase(row.name)}{row.name === data.selected_model ? " (selected)" : ""}</td>
                  <td>{number(row.cv_average_precision_mean, 3)}</td>
                  <td>{number(row.cv_roc_auc_mean, 3)}</td>
                  <td>{percent(row.cv_recall_at_0_5_mean)}</td>
                  <td>{percent(row.cv_precision_at_0_5_mean)}</td>
                  <td>{number(row.cv_f1_at_0_5_mean, 3)}</td>
                  <td>{percent(row.cv_accuracy_at_0_5_mean)}</td>
                  <td>{number(row.fit_seconds, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Held-out confusion matrix" description={`Threshold ${number(test.threshold, 2)}. Scored after the model and threshold were frozen.`}>
          <div className="grid grid-cols-2 gap-2 text-center text-sm">
            <Cell label="True retained" value={matrix.tn} />
            <Cell label="False churn" value={matrix.fp} />
            <Cell label="Missed churn" value={matrix.fn} />
            <Cell label="Caught churn" value={matrix.tp} />
          </div>
          <p className="mt-3 text-sm text-muted">Brier score {number(test.brier, 3)}. Latency {number(test.latency?.milliseconds_per_row, 3)} ms per row.</p>
        </Panel>
        <Panel title="Thresholds" description={data.thresholds.rationale}>
          <p className="text-sm">Medium {number(data.thresholds.medium, 2)} · High / decision {number(data.thresholds.high, 2)}</p>
          <p className="mt-2 text-sm text-muted">Calibration: {data.calibration.reason}</p>
          <p className="mt-2 text-sm text-muted">{data.resampling_experiment.note} Adopted: {data.resampling_experiment.adopted ? "yes" : "no"}.</p>
          <p className="mt-2 text-sm">{data.exclusion_reason}</p>
        </Panel>
        {roc.length > 0 && <Panel title="ROC curve" description="Held-out test set, after model selection."><CurveChart points={roc} xKey="fpr" yKey="tpr" xLabel="False positive rate" yLabel="True positive rate" /></Panel>}
        {pr.length > 0 && <Panel title="Precision-recall curve" description="Held-out test set."><CurveChart points={pr} xKey="recall" yKey="precision" xLabel="Recall" yLabel="Precision" /></Panel>}
        {calibration.length > 0 && <Panel title="Calibration" description="Mean predicted probability against the observed churn rate in each bin."><CurveChart points={calibration} xKey="mean_predicted" yKey="fraction_positive" xLabel="Mean predicted" yLabel="Observed rate" /></Panel>}
        <Panel title="Validation permutation importance" description={data.feature_importance.note}>
          <ImportanceBars rows={data.feature_importance.features} />
        </Panel>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-2xl border border-line bg-surface p-4"><p className="text-xs uppercase text-muted">{label}</p><p className="font-serif text-2xl">{value}</p></div>;
}
function Cell({ label, value }: { label: string; value: number }) {
  return <div className="rounded-xl border border-line p-4"><p className="text-xs text-muted">{label}</p><p className="font-serif text-2xl">{value.toLocaleString()}</p></div>;
}
