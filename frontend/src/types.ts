export interface ChurnRow {
  name: string;
  customers: number;
  churned: number;
  retained: number;
  churn_rate: number | null;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model_error: string | null;
  dataset_loaded: boolean;
  dataset_label: string | null;
  customers: number;
}

export interface Summary {
  dataset_label: string | null;
  total_customers: number;
  observed_churn_count: number;
  observed_retained_count: number;
  observed_churn_rate: number | null;
  predictions_available: boolean;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  unscored_count: number;
  average_predicted_probability: number | null;
  follow_up_required: number;
  contract_churn: ChurnRow[];
  risk_breakdown: { name: string; customers: number }[];
  priority_customers: PriorityCustomer[];
  recent_activity: Activity[];
  time_series_available: boolean;
  time_series_note: string;
  decision_support_warning: string;
  denominator_note: string;
}

export interface PriorityCustomer {
  customer_id: string;
  probability: number | null;
  risk_category: string | null;
  priority_score: number | null;
  contract: string | null;
  tenure: number;
  monthly_charges: number;
  observed_churn: string | null;
}

export interface Activity {
  type: string;
  at: string | null;
  customer_id: string | null;
  summary: string;
}

export interface CustomerItem {
  customer_id: string;
  contract: string | null;
  tenure: number;
  monthly_charges: number;
  total_charges: number | null;
  internet_service: string | null;
  payment_method: string | null;
  paperless_billing: string | null;
  tech_support: string | null;
  observed_churn: string | null;
  probability: number | null;
  risk_category: string | null;
  predicted_class: string | null;
  priority_score: number | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface SegmentMembership {
  id: string;
  name: string;
  definition: string;
  members: number;
}

export interface CustomerPage extends Page<CustomerItem> {
  segment?: SegmentMembership | null;
}

export interface Factor {
  feature: string;
  actual: string | number | null;
  baseline: string | number | null;
  contribution: number;
  direction: string;
  detail: string;
}

export interface Recommendation {
  action: string;
  reason: string;
  effort: string;
  category: string;
}

export interface Prediction {
  customer_id: string | null;
  probability: number | null;
  risk_category: string | null;
  predicted_class: string | null;
  priority_score: number | null;
  factors: Factor[];
  recommendations: Recommendation[];
  reliability: { level: string; message: string };
  explanation_method: string;
  explanation_fallback?: string | null;
  source?: string;
  created_at?: string | null;
  thresholds: { medium: number; high: number; decision: number };
  decision_support_warning: string;
  warnings: string[];
  status?: string;
  errors?: string[];
  row_number?: number;
}

export interface Analytics {
  population: number;
  labeled_customers: number;
  churned: number;
  retained: number;
  churn_rate: number | null;
  denominator_note: string;
  protected_attribute_note: string;
  time_series_note: string;
  by_contract: ChurnRow[];
  by_tenure: ChurnRow[];
  by_monthly_charges: ChurnRow[];
  by_total_charges: ChurnRow[];
  by_internet_service: ChurnRow[];
  by_payment_method: ChurnRow[];
  by_paperless_billing: ChurnRow[];
  by_senior_citizen: ChurnRow[];
  by_tech_support: ChurnRow[];
  by_online_security: ChurnRow[];
  by_gender: ChurnRow[];
  by_partner: ChurnRow[];
  by_dependents: ChurnRow[];
  by_addon_group: ChurnRow[];
  numeric_by_churn: Record<string, Record<string, { mean: number | null; median: number | null; customers: number }>>;
  correlation: { left: string; right: string; correlation: number | null }[];
  predicted: { scored_customers: number; average_probability: number | null; risk_counts: Record<string, number> };
  filters: Record<string, string | number | null>;
  population_before_filters: number;
}

export interface SegmentRule {
  id: string;
  name: string;
  definition: string;
  suggested_action: string;
  customers: number;
  share_of_population: number;
  observed_churned: number;
  observed_churn_rate: number | null;
  average_predicted_probability: number | null;
  average_monthly_charges: number | null;
  average_tenure: number | null;
  primary_customers: number;
}

export interface ClusterSegment {
  label: string;
  customers: number;
  observed_churned: number;
  observed_churn_rate: number | null;
  average_tenure: number;
  average_monthly_charges: number;
  average_predicted_probability: number | null;
}

export interface Campaign {
  id: number;
  customer_id: string;
  campaign_name: string;
  action: string;
  reason: string;
  effort: string;
  category: string;
  status: string;
  priority_score: number;
  notes: string | null;
  updated_at: string | null;
  outcomes_available: boolean;
  outcomes_note: string;
}

export interface Scenario {
  name: string;
  success_rate: number;
  success_rate_capped_at_one: boolean;
  customers_targeted: number;
  expected_churners: number;
  revenue_exposure: number;
  contribution_exposure: number;
  expected_retained_customers: number;
  estimated_contribution_benefit: number;
  campaign_cost: number;
  estimated_net_benefit: number;
  break_even_success_rate: number | null;
}

export interface Impact {
  disclaimer: string;
  assumptions_note: string;
  outcomes_available: boolean;
  outcomes_note: string;
  formulas: string[];
  assumptions: {
    avg_monthly_revenue: number;
    contribution_margin: number;
    campaign_cost: number;
    success_rate: number;
    horizon_months: number;
    intervention_capacity: number | null;
  };
  scenarios: Scenario[];
  predictions_available: boolean;
  reference: {
    customers_total: number;
    customers_at_risk_before_capacity: number;
    data_average_monthly_charges: number | null;
    observed_churn_rate: number | null;
  };
}

export interface Settings {
  medium_threshold: number;
  high_threshold: number;
  thresholds_customized: boolean;
  avg_monthly_revenue: number | null;
  contribution_margin: number;
  campaign_cost: number;
  success_rate: number;
  horizon_months: number;
  intervention_capacity: number | null;
  decision_support_warning: string;
  excluded_features_reason: string;
  threshold_note: string;
}

export interface ModelInfo {
  model_loaded: boolean;
  model_error: string | null;
  selected_model: string | null;
  trained_at: string | null;
  selection_reason: string | null;
  feature_spec?: {
    numeric: Record<string, { min: number; max: number; median: number }>;
    categorical: Record<string, string[]>;
  };
  excluded_features?: string[];
  exclusion_reason?: string;
}

export interface ComparisonRow {
  name: string;
  status: string;
  cv_average_precision_mean: number;
  cv_roc_auc_mean: number;
  cv_recall_at_0_5_mean: number;
  cv_precision_at_0_5_mean: number;
  cv_f1_at_0_5_mean: number;
  cv_accuracy_at_0_5_mean: number;
  fit_seconds: number;
}

export interface Performance {
  selected_model: string;
  selection_reason: string;
  comparison_scope: string;
  comparison: ComparisonRow[];
  calibration: { applied: boolean; reason: string; validation_brier_uncalibrated?: number; validation_brier_calibrated?: number | null };
  thresholds: { medium: number; high: number; decision: number; rationale: string };
  validation_metrics: MetricBlock;
  test_metrics: MetricBlock & {
    curves?: {
      roc_curve: { fpr: number[]; tpr: number[] };
      pr_curve: { recall: number[]; precision: number[] };
      calibration_curve: { mean_predicted: number[]; fraction_positive: number[] };
    };
    latency?: { milliseconds_per_row: number };
  };
  feature_importance: { features: { feature: string; importance: number; std: number }[]; note: string };
  split: { n_train: number; n_validation: number; n_test: number; test_churn_rate: number };
  excluded_features: string[];
  exclusion_reason: string;
  resampling_experiment: {
    class_weight_balanced_average_precision: number;
    random_oversample_average_precision: number;
    adopted: boolean;
    applied_to_deployed_model: boolean;
    note: string;
  };
}

export interface MetricBlock {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  average_precision: number;
  brier: number;
  threshold: number;
  label?: string;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface GlobalExplanation {
  feature_importance: Performance["feature_importance"];
  associations: {
    training_churn_rate: number;
    caveat: string;
    categorical: { feature: string; levels: { level: string; customers: number; churn_rate: number }[] }[];
    numeric: { feature: string; correlation_with_churn: number | null; note: string }[];
  } | null;
  limitation: string;
  note: string;
  exclusion_reason: string;
}

export interface QualityResponse {
  available: boolean;
  message?: string;
  dataset_label?: string;
  filename?: string;
  imported_at?: string | null;
  stored_customers?: number;
  quality?: {
    rows_raw: number;
    rows_cleaned: number;
    missing_values: Record<string, number>;
    duplicate_customer_ids: number;
    class_counts: Record<string, number>;
    observed_churn_rate: number | null;
    outliers: Record<string, { flagged: number; note: string }>;
    cleaning_notes: string[];
    outlier_policy: string;
    invalid_numeric_values: Record<string, number>;
  };
}
