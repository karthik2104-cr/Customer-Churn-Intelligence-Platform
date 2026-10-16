import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Shell } from "../components/Shell";
import { AnalyticsPage } from "../pages/AnalyticsPage";
import { BatchPage } from "../pages/BatchPage";
import { CustomerPage } from "../pages/CustomerPage";
import { CustomersPage } from "../pages/CustomersPage";
import { OverviewPage } from "../pages/OverviewPage";
import { PredictPage } from "../pages/PredictPage";
import { SegmentsPage } from "../pages/SegmentsPage";

const summary = {
  dataset_label: "synthetic",
  total_customers: 12,
  observed_churn_count: 4,
  observed_retained_count: 8,
  observed_churn_rate: 0.333,
  predictions_available: true,
  high_risk_count: 2,
  medium_risk_count: 3,
  low_risk_count: 7,
  unscored_count: 0,
  average_predicted_probability: 0.28,
  follow_up_required: 1,
  contract_churn: [{ name: "Month-to-month", customers: 6, churned: 3, retained: 3, churn_rate: 0.5 }],
  risk_breakdown: [{ name: "High", customers: 2 }, { name: "Low", customers: 7 }],
  priority_customers: [],
  recent_activity: [],
  time_series_available: false,
  time_series_note: "No event dates are available.",
  decision_support_warning: "This is decision support.",
  denominator_note: "Rates use labeled customers.",
};

function jsonResponse(body: unknown, ok = true) {
  return Promise.resolve({
    ok,
    status: ok ? 200 : 500,
    statusText: ok ? "OK" : "Error",
    text: () => Promise.resolve(JSON.stringify(body)),
    json: () => Promise.resolve(body),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("dashboard pages", () => {
  it("renders the overview from the summary payload", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse(summary)));
    render(<MemoryRouter><OverviewPage /></MemoryRouter>);
    expect(await screen.findByText("Observed churn rate")).toBeInTheDocument();
    expect(screen.getByText("33.3%")).toBeInTheDocument();
    expect(screen.getByText("This is decision support.")).toBeInTheDocument();
  });

  it("shows an error when the overview request fails", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ message: "Database unavailable" }, false)));
    render(<MemoryRouter><OverviewPage /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("Database unavailable");
  });

  it("requests analytics again when a filter changes", async () => {
    const analytics = {
      population: 10,
      labeled_customers: 10,
      churned: 2,
      retained: 8,
      churn_rate: 0.2,
      denominator_note: "Labeled customers in the filter.",
      protected_attribute_note: "Descriptive only.",
      time_series_note: "No dates.",
      by_contract: [],
      by_tenure: [],
      by_monthly_charges: [],
      by_total_charges: [],
      by_internet_service: [],
      by_payment_method: [],
      by_paperless_billing: [],
      by_senior_citizen: [],
      by_tech_support: [],
      by_online_security: [],
      by_gender: [],
      by_partner: [],
      by_dependents: [],
      by_addon_group: [],
      numeric_by_churn: {},
      correlation: [],
      predicted: { scored_customers: 0, average_probability: null, risk_counts: {} },
      filters: {},
      population_before_filters: 10,
    };
    const fetchMock = vi.fn((_input: RequestInfo | URL) => jsonResponse(analytics));
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><AnalyticsPage /></MemoryRouter>);
    expect(await screen.findByText("How churn differs across the base")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Contract"), "Month-to-month");
    await waitFor(() => {
      const urls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(urls.some((url) => url.includes("contract=Month-to-month"))).toBe(true);
    });
  });

  it("blocks a prediction when tenure is empty", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ model_loaded: true, model_error: null, feature_spec: { categorical: {} } })));
    render(<MemoryRouter><PredictPage /></MemoryRouter>);
    const tenure = await screen.findByLabelText("Tenure (months)");
    await userEvent.clear(tenure);
    await userEvent.click(screen.getByRole("button", { name: "Score customer" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Tenure must be zero or greater.");
  });

  it("shows a scored prediction", async () => {
    const fetchMock = vi.fn((url: string) => {
      if (String(url).includes("/api/predictions/single")) {
        return jsonResponse({
          customer_id: "SYN-1",
          probability: 0.62,
          risk_category: "High",
          predicted_class: "Yes",
          priority_score: 71,
          factors: [],
          recommendations: [{ action: "Offer a longer contract", reason: "Month-to-month and elevated risk.", effort: "medium", category: "contract" }],
          reliability: { level: "typical", message: "The probability is not a certainty." },
          explanation_method: "occlusion",
          thresholds: { medium: 0.3, high: 0.5, decision: 0.5 },
          decision_support_warning: "Decision support only.",
          warnings: [],
        });
      }
      return jsonResponse({ model_loaded: true, model_error: null, feature_spec: { categorical: {} } });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><PredictPage /></MemoryRouter>);
    await screen.findByLabelText("Tenure (months)");
    await userEvent.click(screen.getByRole("button", { name: "Score customer" }));
    expect(await screen.findByText("62.0%")).toBeInTheDocument();
    expect(screen.getByText("Offer a longer contract")).toBeInTheDocument();
  });

  it("asks for a file before batch scoring", async () => {
    render(<MemoryRouter><BatchPage /></MemoryRouter>);
    await userEvent.click(screen.getByRole("button", { name: "Score file" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose a CSV file first.");
  });

  it("renders a customer profile from the stored record", async () => {
    const detail = {
      profile: { customerID: "5419-JPRRN", Contract: "Month-to-month", tenure: 1, MonthlyCharges: 70, TotalCharges: 70, Churn: "Yes", InternetService: "Fiber optic", PaymentMethod: "Electronic check" },
      profile_note: "Descriptive fields are not used for scoring.",
      prediction: {
        customer_id: "5419-JPRRN",
        probability: 0.819,
        risk_category: "High",
        predicted_class: "Yes",
        priority_score: 80,
        factors: [{ feature: "Contract", actual: "Month-to-month", baseline: "Two year", contribution: 0.12, direction: "higher_predicted_churn", detail: "Contract" }],
        recommendations: [{ action: "Prioritize a term-contract conversation", reason: "Month-to-month and elevated risk.", effort: "medium", category: "contract" }],
        reliability: { level: "typical", message: "Probability is not certainty." },
        explanation_method: "occlusion",
        thresholds: { medium: 0.1, high: 0.16, decision: 0.16 },
        decision_support_warning: "Decision support only.",
        warnings: [],
        created_at: "2026-10-16T12:00:00Z",
      },
      history: [],
      campaigns: [],
      explanation_limitation: "Association, not causation.",
      decision_support_warning: "Decision support only.",
    };
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse(detail)));
    render(
      <MemoryRouter initialEntries={["/customers/5419-JPRRN"]}>
        <Routes>
          <Route path="/customers/:customerId" element={<CustomerPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("heading", { name: "5419-JPRRN" })).toBeInTheDocument();
    expect(screen.getByText("81.9%")).toBeInTheDocument();
    expect(screen.getByText("Contract")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record follow-up" })).toBeInTheDocument();
    expect(screen.getByText("No follow-up has been recorded. Recording one does not create a measured save rate.")).toBeInTheDocument();
  });

  it("links a segment to the exact membership filter", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        jsonResponse({
          high_charge_cutoff: 80,
          cutoff_note: "75th percentile.",
          dataset_label: "synthetic",
          cluster_note: "Clusters were fit without churn.",
          clusters: [],
          rules: [
            {
              id: "low_risk_expansion",
              name: "Low-risk expansion",
              definition: "Low risk, tenure of at least 12 months, internet service, and two or fewer add-ons.",
              suggested_action: "Consider an add-on conversation.",
              customers: 0,
              share_of_population: 0,
              observed_churned: 0,
              observed_churn_rate: null,
              average_predicted_probability: null,
              average_monthly_charges: null,
              average_tenure: null,
              primary_customers: 0,
            },
          ],
        }),
      ),
    );
    render(<MemoryRouter><SegmentsPage /></MemoryRouter>);
    const link = await screen.findByRole("link", { name: "View exact members" });
    expect(link).toHaveAttribute("href", "/customers?segment=low_risk_expansion&page=1");
  });

  it("requests the segment filter and shows the matching total", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL) =>
      jsonResponse({
        items: [],
        total: 0,
        page: 1,
        page_size: 25,
        segment: {
          id: "low_risk_expansion",
          name: "Low-risk expansion",
          definition: "Low risk and room for an add-on.",
          members: 0,
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(
      <MemoryRouter initialEntries={["/customers?segment=low_risk_expansion&page=1"]}>
        <CustomersPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText(/0 customers match every condition/)).toBeInTheDocument();
    await waitFor(() => {
      const urls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(urls.some((url) => url.includes("segment=low_risk_expansion"))).toBe(true);
    });
  });

  it("keeps a navigation control available for narrow layouts", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ status: "ok", model_loaded: false, model_error: null, dataset_loaded: false, dataset_label: null, customers: 0 })));
    render(
      <MemoryRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<p>Home body</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("button", { name: "Open navigation" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
  });
});
