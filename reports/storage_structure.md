# Storage Structure Documentation
**RecoMart Pipeline — Task 3**
Dataset: Amazon Reviews 2023 — Video Games
Generated: 2026-07-07 02:58 UTC

## Data Lake Layout

| Path | Content | Format | Partition Key |
|------|---------|--------|---------------|
| `data/raw/interactions/` | User-item ratings | CSV | Ingest timestamp |
| `data/raw/products/` | Product metadata catalog | CSV | Ingest timestamp |
| `data/raw/logs/` | Ingestion audit trail | CSV (append-only) | — |
| `data/processed/` | Cleaned + merged datasets | CSV | Pipeline run |

## Source Files (place before running)
| File | Location | Description |
|------|----------|-------------|
| `Video_Games.jsonl` | `data/raw/interactions/` | Amazon 2023 ratings JSONL |
| `meta_Video_Games.jsonl` | `data/raw/products/` | Amazon 2023 metadata JSONL |

## JSONL Field Mapping

### Ratings (Video_Games.jsonl)
| Raw Field | Pipeline Field | Description |
|-----------|---------------|-------------|
| `user_id` | `userId` | Reviewer ID |
| `parent_asin` | `productId` | Product ASIN |
| `rating` | `rating` | Star rating 1.0–5.0 |
| `timestamp` | `timestamp` | Unix epoch (ms → s) |

### Metadata (meta_Video_Games.jsonl)
| Raw Field | Pipeline Field | Description |
|-----------|---------------|-------------|
| `parent_asin` | `productId` | Product ASIN |
| `title` | `title` | Product title |
| `price` | `price` | Numeric price (USD) |
| `store` | `brand` | Brand/store name |
| `categories` | `category` | Category path string |
| `description` | `description` | Text (truncated 300 chars) |

## Retention Policy
- Raw JSONL source files: retained locally (not uploaded to Drive — too large)
- Sampled CSV outputs: uploaded to Google Drive `02_dataset/raw/`
- Audit log: append-only, never deleted
- Processed data: versioned via DVC (Task 8)
