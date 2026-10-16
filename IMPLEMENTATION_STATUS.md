# Implementation status

Updated 17 October 2026 after the quality audit. The navy-and-teal interface and the calibrated random forest were kept.

## Completed

- Data loading, schema checks, blank `TotalCharges` handling, and a quality report. Eleven tenure-0 rows with a blank total are kept with total charges set to 0.
- EDA for contract, tenure, charges, internet, payment, paperless billing, tech support, online security, add-ons, and descriptive demographic charts. Counts and rates are separate. There is no monthly trend because the file has no event date.
- Leakage-safe comparison of logistic regression, decision tree, random forest, histogram gradient boosting, and XGBoost. Random forest was selected. Sigmoid calibration was kept. Held-out metrics are in `ml/artifacts/metadata.json` and `docs/ml-methodology.md`.
- FastAPI app with SQLite, prediction, batch upload, explanations, segments, campaigns, impact scenarios, and settings.
- React command center for every primary navigation item, connected to that API. The visual system is navy and teal, with a persisted light theme and dark theme.
- Segment membership uses one classifier for the card counts and `GET /api/customers?segment=`. The directory lists customers who satisfy every condition of that rule.
- Dedicated customer profile at `/customers/{id}`: subscription details from the stored row, predicted probability, risk band, occlusion factors, retention actions, prediction history, and the persisted follow-up.
- Pytest and Vitest coverage for the pipeline, API, and main UI states, including exact segment membership, empty-segment behavior, upload filename handling, and the profile page.
- Docker Compose and GitHub Actions files. Compose configuration parses. The Docker daemon and remote CI were not run in this session.

## Commands that were run

Training, from the repository root, finished with exit code 0 in about 38 minutes on 16 October 2026:

```powershell
.\.venv\Scripts\python.exe -m ml.training.train --csv "data/raw/Telco-Customer-Churn.csv" --output "ml/artifacts" --cleaned "data/processed/telco_cleaned.csv"
```

Selected random forest. Held-out average precision 0.6521, recall 0.8610, F1 0.6070 at threshold 0.16. Those values were read again from `GET /api/models/performance` on 17 October 2026 and still match.

Backend tests rerun on 17 October 2026, from the repository root:

```text
.\.venv\Scripts\python.exe -m pytest
24 passed, 4 warnings in 77.77s
```

The warnings are a Starlette note about `httpx` and a SHAP colormap deprecation. They do not fail the suite. The tests use an isolated database, not `data/app.db`.

Frontend, from `frontend/`:

```text
npm run typecheck
npm test
npm run build
```

The same three commands were rerun on 17 October 2026. Typecheck passed. Vitest: 10 passed. Production build completed (`vite build`, built in 11.74s, exit code 0). The bundle is one large JavaScript file; that is a size warning, not a failed build.

Docker:

```text
docker version
docker compose config
```

Docker client 29.7.2 is installed. The daemon was not running (`npipe:////./pipe/dockerDesktopLinuxEngine` was missing), so images were not built and containers were not started. `docker compose config` parsed. It publishes the UI at `127.0.0.1:5173` and the API at `127.0.0.1:8000`.

The push of commit `873a743` triggered GitHub Actions run 1 on `master`. Both jobs completed successfully: pytest, and the frontend `npm ci`, typecheck, Vitest, and production build. The run is https://github.com/karthik2104-cr/Customer-Churn-Intelligence-Platform/actions/runs/38966344163.

## Browser verification

On 17 October 2026 the API was started with `uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`. `netstat` showed `127.0.0.1:8000` listening. The UI was started at `http://localhost:5180` with `VITE_API_PROXY=http://127.0.0.1:8000` because port 5173 was already used by another local app. The documented defaults remain 8000 and 5173.

Checked in the browser on 17 October 2026:

- Overview shows 7,043 customers, 26.5% observed churn (1,869 / 5,174), and 3,471 high-risk scores. Light and dark themes both render that page.
- Analytics filter for month-to-month updates the denominator to 3,875 customers, 1,655 churned, 42.7%.
- Search for `5419-JPRRN` returns one directory row. Open loads that customer's profile: month-to-month, 1 month, $101.45, observed churn Yes, predicted probability 81.9%, high risk, baseline occlusion factors, and follow-up status Contacted.
- Explainability for the same id returns 81.9% and the same occlusion method.
- Opening "View exact members" for early-tenure elevated risk goes to `/customers?segment=early_tenure_elevated_risk&page=1` and shows 1,383 customers, matching the segment card. Adding Two year to the month-to-month elevated segment shows 0 of 3,531 members. A read-only API check matched `total`, `segment.members`, and the card count for all seven rules. Primary assignments sum to 7,043.
- Segments show rule groups separately from k-means profiles. The cluster note says churn was not a clustering feature.
- Campaign status for `5419-JPRRN` is still Contacted after an API restart.
- Business impact loads the stored assumptions (revenue 64.77, cost 25, margin 0.70, success 0.30, horizon 12) and labels the results as estimates.
- Model page shows Random Forest, held-out average precision 0.652, and confusion counts 670 / 365 / 52 / 322.
- The prediction form with an empty tenure shows "Tenure must be zero or greater" and does not call the API. Batch scoring without a file shows "Choose a CSV file first." A successful live score was not submitted, because those endpoints persist rows into the demo database. Scoring itself is covered by the isolated API tests.
- At 390px the menu button is visible (`display: block`), the sidebar transform is `matrix(1, 0, 0, 1, -288, 0)`, and the document width is 390. Opening the menu shows the close control and overlay. The theme toggle switches `html` to `dark` and back to light.

Screenshots from that session are in `docs/screenshots/`.

## Not done here

- Docker images were not built and Compose was not started, because the Docker daemon was not running.
- GitHub Actions run 1, for commit `873a743`, completed successfully. Docker images are still unverified.
- There is no login. Do not expose the API publicly.
- The calibrated random forest is explained with baseline occlusion. SHAP is implemented for estimators it supports, and it is discarded when the values do not match the probability.
- Campaign outcomes remain unavailable. The file has no save-rate field.
- Histogram gradient boosting took 1,528 seconds in the full search because early stopping is off and each candidate uses 180 iterations.
