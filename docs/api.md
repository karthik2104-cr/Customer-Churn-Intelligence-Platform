# API

Base path: `/api`. Interactive schema: `http://127.0.0.1:8000/docs` when the API is running on the documented local bind. Do not publish the process without authentication.

Errors use `{ "code", "message" }`. Validation failures also return an `errors` list with `type`, `loc`, and `msg` only. Submitted values, exception context, and stack traces are not returned.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process, model, and dataset status |
| GET | `/dashboard/summary` | Overview counts, separating observed churn from predicted risk |
| GET | `/analytics/churn` | Filtered churn rates and counts |
| GET | `/analytics/segments` | Rule segments and cluster profiles |
| GET | `/customers` | Paginated, filterable customer table |
| GET | `/customers/{customer_id}` | Profile, latest score, history, campaigns |
| POST | `/predictions/single` | One customer score, factors, and actions |
| POST | `/predictions/batch` | CSV upload; invalid rows are returned, not dropped |
| GET | `/predictions/batches/{batch_id}` | Paginated batch results |
| GET | `/predictions/batches/{batch_id}/download` | CSV download with formula-injection protection |
| GET | `/models` | Loaded model and feature specification |
| GET | `/models/performance` | Comparison and held-out metrics from `metadata.json` |
| POST | `/models/reload` | Reload the artifact and rescore stored customers |
| GET | `/explanations/global` | Permutation importance and training associations |
| GET | `/explanations/{customer_id}` | Local explanation for a stored customer |
| GET | `/campaigns` | Follow-up records |
| POST | `/campaigns` | Create a follow-up |
| PATCH | `/campaigns/{campaign_id}` | Update status or notes |
| POST | `/campaigns/suggest` | Create follow-ups for elevated-risk customers |
| GET | `/business-impact` | Conservative, base, and optimistic scenarios |
| GET, PUT | `/settings` | Thresholds and financial assumptions |
| GET | `/data-quality` | Latest import quality report |
| POST | `/data/import` | CSV import, schema-checked |

Uploads must be `.csv` and are limited by `MAX_UPLOAD_MB` (default 10). A request body larger than that limit plus 1 MB of multipart framing is rejected before the CSV is parsed. The stored filename is a basename, not a filesystem path. The model file is loaded only from `MODEL_PATH`. Uploads are never unpickled.

Customer list filters include `search`, `contract`, `internet_service`, `risk`, `observed_churn`, `tenure_min`, `tenure_max`, `monthly_min`, `monthly_max`, `segment`, `sort`, and `order`. `segment` is a rule id from `/analytics/segments`. The returned customers satisfy every condition of that rule. `total` is the count after the other directory filters. `segment.members` is the full rule count. With no other filters, those two counts match.

Analytics filters include `contract`, `internet_service`, `payment_method`, `paperless_billing`, `tech_support`, `senior_citizen`, `tenure_min`, and `tenure_max`.
