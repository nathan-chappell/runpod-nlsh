# Live Served Demo - 2026-04-25

This note captures the live checks against the served Runpod endpoint and the follow-up real shell execution demo in the local playground workspace.

Supporting artifacts for this note were copied under:

- [live-served-demo-2026-04-25-assets/](live-served-demo-2026-04-25-assets/)

## Setup

| Item | Value |
| --- | --- |
| Served endpoint | `https://u4e8stvpi9nd6a-8000.proxy.runpod.net/` |
| OpenAI-compatible base URL | `https://u4e8stvpi9nd6a-8000.proxy.runpod.net/v1` |
| Served model | `microsoft/Phi-4-mini-instruct:nlsh-phi4-ft` |
| Local execution workspace | `tmp/playground-workspace` |

The service responded successfully to:

- `/model_info`
- `/v1/models`
- `/v1/chat/completions`

## Live Probe

Saved artifact:

- [probe-latest.json](live-served-demo-2026-04-25-assets/probe-latest.json)

Summary:

| Metric | Value |
| --- | --- |
| Samples | `10` |
| Exact matches | `8/10` |
| Compile-valid | `10/10` |
| Transport / parse failures | `0` |

The two misses are short enough to show inline.

### Miss 1

Prompt:

```text
merge the signed PDFs into signed_packet.pdf
```

Expected:

```json
{"kind":"clarification","question":"Which PDF files should I merge into signed_packet.pdf?"}
```

Actual:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"","glob":"*signed*.pdf"},{"kind":"pdf_merge","output_file":"signed_packet.pdf"}]}
```

### Miss 2

Prompt:

```text
find every PDF under ./board with executed in the path
```

Expected:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./board","glob":"*executed*.pdf"}]}
```

Actual:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./board","glob":"*executed*.pdf","max_depth":2}]}
```

So the live probe tells the same story as the offline evals: the model is excellent at producing legal plans, but it still over-assumes on some underspecified merge prompts.

## Real Execution Demo

Saved artifact:

- [live-run-report.json](live-served-demo-2026-04-25-assets/live-run-report.json)

Overall result:

| Metric | Value |
| --- | --- |
| Executed successfully | `8` |
| Clarifications | `0` |
| Other failures | `0` |

Prompts run:

| Category | Prompt | Result |
| --- | --- | --- |
| `csv_to_json` | `convert orders.csv to JSON in orders.json` | executed |
| `json_filter` | `from users.json keep rows where role is not guest into member_users.json` | executed |
| `json_select_fields` | `from tickets.json keep only ticket_id priority and owner into ticket_queue.json` | executed |
| `json_sort` | `sort invoices.json by total descending into invoices_sorted.json` | executed |
| `json_group_count` | `count incidents.json rows per severity into incidents_by_severity.json` | executed |
| `pdf_merge` | `merge contract_a.pdf and contract_b.pdf into contract_packet.pdf` | executed |
| `pdf_extract_pages` | `extract pages 2 through 3 from handbook.pdf into handbook_excerpt.pdf` | executed |
| `find_files` | `find every PDF under ./contracts with executed in the path` | executed |

## Playground Inputs

Source CSV:

- [orders.csv](live-served-demo-2026-04-25-assets/orders.csv)

Rendered as a table:

| order_id | status | total |
| --- | --- | --- |
| `1001` | `paid` | `19.95` |
| `1002` | `pending` | `42.00` |
| `1003` | `paid` | `10.50` |

## Observed Outputs

### `orders.json`

- [orders.json](live-served-demo-2026-04-25-assets/orders.json)

```json
[
  {
    "order_id": "1001",
    "status": "paid",
    "total": "19.95"
  },
  {
    "order_id": "1002",
    "status": "pending",
    "total": "42.00"
  },
  {
    "order_id": "1003",
    "status": "paid",
    "total": "10.50"
  }
]
```

### `member_users.json`

- [member_users.json](live-served-demo-2026-04-25-assets/member_users.json)

```json
[
  {
    "user_id": 1,
    "role": "admin",
    "email": "alice@example.com"
  },
  {
    "user_id": 3,
    "role": "member",
    "email": "cora@example.com"
  }
]
```

### `ticket_queue.json`

- [ticket_queue.json](live-served-demo-2026-04-25-assets/ticket_queue.json)

```json
[
  {
    "ticket_id": "T-100",
    "priority": "high",
    "owner": "alice"
  },
  {
    "ticket_id": "T-101",
    "priority": "medium",
    "owner": "bob"
  }
]
```

### `invoices_sorted.json`

- [invoices_sorted.json](live-served-demo-2026-04-25-assets/invoices_sorted.json)

```json
[
  {
    "invoice_id": "INV-2",
    "total": 99.5
  },
  {
    "invoice_id": "INV-3",
    "total": 48.2
  },
  {
    "invoice_id": "INV-1",
    "total": 15
  }
]
```

### `incidents_by_severity.json`

- [incidents_by_severity.json](live-served-demo-2026-04-25-assets/incidents_by_severity.json)

```json
[
  {
    "severity": "high",
    "count": 2
  },
  {
    "severity": "low",
    "count": 1
  }
]
```

### `find_files` output

The `find_files` prompt returned:

```text
./contracts/executed_alpha.pdf
./contracts/executed_beta.pdf
```

### PDF outputs

- [contract_packet.pdf](live-served-demo-2026-04-25-assets/contract_packet.pdf)
- [handbook_excerpt.pdf](live-served-demo-2026-04-25-assets/handbook_excerpt.pdf)

## Output Validation

### JSON outputs

| File | Validation |
| --- | --- |
| `orders.json` | parsed successfully, `3` rows |
| `member_users.json` | parsed successfully, roles reduced to `admin`, `member` |
| `ticket_queue.json` | parsed successfully, keys reduced to `ticket_id`, `priority`, `owner` |
| `invoices_sorted.json` | parsed successfully, totals sorted descending: `99.5`, `48.2`, `15` |
| `incidents_by_severity.json` | parsed successfully, grouped counts `high -> 2`, `low -> 1` |

### PDF outputs

| File | Validation |
| --- | --- |
| `contract_packet.pdf` | `qpdf --check` passed, page count `2` |
| `handbook_excerpt.pdf` | `qpdf --check` passed, page count `2` |

## Interpretation

This run is a strong practical result for the project:

- the served model is reachable and stable through the Runpod proxy
- the runtime planner prompt shape works against the served endpoint
- the model can produce compile-ready plans on live prompts
- those plans can be executed against real files with the expected outputs

The remaining weakness is still the same one shown in offline evals and the 10-sample live probe:

- underspecified merge prompts can still trigger overconfident planning instead of clarification

But for fully specified prompts in the supported action set, the end-to-end story now works: plan, compile, confirm, execute, validate.
