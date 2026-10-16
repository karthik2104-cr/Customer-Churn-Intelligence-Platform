import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { BatchPage } from "./pages/BatchPage";
import { CampaignsPage } from "./pages/CampaignsPage";
import { CustomerPage } from "./pages/CustomerPage";
import { CustomersPage } from "./pages/CustomersPage";
import { DataPage } from "./pages/DataPage";
import { ExplainPage } from "./pages/ExplainPage";
import { ImpactPage } from "./pages/ImpactPage";
import { ModelPage } from "./pages/ModelPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PredictPage } from "./pages/PredictPage";
import { SegmentsPage } from "./pages/SegmentsPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<OverviewPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="customers" element={<CustomersPage />} />
        <Route path="customers/:customerId" element={<CustomerPage />} />
        <Route path="predict" element={<PredictPage />} />
        <Route path="batch" element={<BatchPage />} />
        <Route path="segments" element={<SegmentsPage />} />
        <Route path="campaigns" element={<CampaignsPage />} />
        <Route path="models" element={<ModelPage />} />
        <Route path="explain" element={<ExplainPage />} />
        <Route path="impact" element={<ImpactPage />} />
        <Route path="data" element={<DataPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
