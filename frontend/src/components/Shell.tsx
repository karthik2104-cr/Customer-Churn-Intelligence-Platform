import { FormEvent, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Calculator,
  Database,
  FileSpreadsheet,
  Layers,
  LayoutDashboard,
  Megaphone,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  Sun,
  UserRound,
  Users,
  X,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import type { Health } from "../types";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/analytics", label: "Customer Analytics", icon: BarChart3 },
  { to: "/customers", label: "Customer Profiles", icon: Users },
  { to: "/predict", label: "Churn Prediction", icon: UserRound },
  { to: "/batch", label: "Batch Prediction", icon: FileSpreadsheet },
  { to: "/segments", label: "Customer Segmentation", icon: Layers },
  { to: "/campaigns", label: "Retention Campaigns", icon: Megaphone },
  { to: "/models", label: "Model Performance", icon: Activity },
  { to: "/explain", label: "Explainability", icon: Sparkles },
  { to: "/impact", label: "Business Impact", icon: Calculator },
  { to: "/data", label: "Data Management", icon: Database },
  { to: "/settings", label: "Settings", icon: Settings },
];

function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem("churn-theme") === "dark");
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("churn-theme", dark ? "dark" : "light");
  }, [dark]);
  return { dark, toggle: () => setDark((value) => !value) };
}

function pageTitle(pathname: string): { section: string; title: string } {
  if (pathname.startsWith("/customers/")) return { section: "Customer Profiles", title: "Customer profile" };
  const match = NAV.find((item) => (item.to === "/" ? pathname === "/" : pathname.startsWith(item.to)));
  return { section: "Churn Intelligence", title: match?.label || "Overview" };
}

export function Shell() {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("churn-sidebar") === "collapsed");
  const [refreshKey, setRefreshKey] = useState(0);
  const [query, setQuery] = useState("");
  const theme = useTheme();
  const health = useApi<Health>("/api/health");
  const location = useLocation();
  const navigate = useNavigate();
  const current = pageTitle(location.pathname);

  useEffect(() => {
    localStorage.setItem("churn-sidebar", collapsed ? "collapsed" : "open");
  }, [collapsed]);

  function search(event: FormEvent) {
    event.preventDefault();
    const next = new URLSearchParams();
    if (query.trim()) next.set("search", query.trim());
    next.set("page", "1");
    navigate(`/customers?${next.toString()}`);
  }

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-72 transform flex-col bg-[var(--sidebar)] text-[rgb(var(--sidebar-ink))] transition lg:translate-x-0 ${
          collapsed ? "lg:w-20" : "lg:w-72"
        } ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex items-center justify-between gap-2 px-4 py-5">
          <div className={collapsed ? "lg:hidden" : ""}>
            <p className="font-serif text-xl leading-none">Churn Intelligence</p>
            <p className="mt-1 text-[11px] uppercase tracking-[0.16em] text-white/60">Decision support</p>
          </div>
          <button type="button" className="lg:hidden" aria-label="Close navigation" onClick={() => setOpen(false)}>
            <X />
          </button>
        </div>
        <nav aria-label="Primary" className="flex-1 space-y-1 overflow-y-auto px-3 pb-4">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              title={item.label}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex min-h-10 items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                  isActive ? "bg-white/10 text-white" : "text-white/75 hover:bg-white/5 hover:text-white"
                }`
              }
            >
              <item.icon size={16} aria-hidden="true" />
              <span className={collapsed ? "lg:sr-only" : ""}>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <button
          type="button"
          className="m-3 hidden min-h-10 items-center gap-2 rounded-lg px-3 py-2 text-sm text-white/70 hover:bg-white/5 lg:flex"
          onClick={() => setCollapsed((value) => !value)}
          aria-pressed={collapsed}
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          <span className={collapsed ? "sr-only" : ""}>{collapsed ? "Expand sidebar" : "Collapse sidebar"}</span>
        </button>
      </aside>
      {open && <button type="button" className="fixed inset-0 z-20 bg-black/50 lg:hidden" aria-label="Close menu overlay" onClick={() => setOpen(false)} />}
      <div className={collapsed ? "lg:pl-20" : "lg:pl-72"}>
        <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-line bg-canvas/95 px-4 py-3 backdrop-blur md:px-8">
          <button type="button" className="rounded-lg border border-line p-2 lg:hidden" aria-label="Open navigation" onClick={() => setOpen(true)}>
            <Menu size={18} />
          </button>
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.14em] text-muted">{current.section}</p>
            <p className="truncate text-sm font-semibold">{current.title}</p>
          </div>
          <form className="order-last flex min-w-[16rem] flex-1 items-center gap-2 sm:order-none" onSubmit={search}>
            <label className="relative block w-full">
              <span className="sr-only">Search customers by id</span>
              <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" aria-hidden="true" />
              <input
                className="w-full rounded-lg border border-line bg-surface py-2 pl-9 pr-3 text-sm"
                placeholder="Search customer id"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <button type="submit" className="min-h-10 rounded-lg border border-line bg-surface px-3 text-sm font-semibold">Search</button>
          </form>
          <p className="hidden text-xs text-muted xl:block">
            {health.loading && "Checking the API"}
            {health.error && health.error}
            {health.data && (
              <>
                {health.data.dataset_label || "Dataset not loaded"}
                {` · ${health.data.customers.toLocaleString()}`}
                {health.data.model_loaded ? " · model loaded" : " · model not loaded"}
              </>
            )}
          </p>
          <button type="button" className="rounded-lg border border-line bg-surface p-2" aria-label="Refresh page data" onClick={() => setRefreshKey((value) => value + 1)}>
            <RefreshCw size={16} />
          </button>
          <button type="button" className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm" onClick={theme.toggle} aria-label="Toggle color theme">
            {theme.dark ? <Sun size={16} /> : <Moon size={16} />}
            <span className="hidden sm:inline">{theme.dark ? "Light" : "Dark"}</span>
          </button>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 md:px-8">
          <Outlet key={refreshKey} />
        </main>
      </div>
    </div>
  );
}
