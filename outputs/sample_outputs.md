# StayChat RAG — Sample Outputs

Pre-generated outputs for the three mandatory evaluation queries.
Produced by running `python main.py --mode demo` with the hotel_documents.json
dataset and llama3-8b-8192 via Groq API.

---

## Query 1: Which hotels have free WiFi and complimentary breakfast?

### Retrieved Chunks (top 5):

| # | chunk_id | hotel | category | score | text preview |
|---|----------|-------|----------|-------|--------------|
| 1 | hotel_010_chunk_0 | Sunrise Boutique Resort | amenities | 0.7821 | "guests at sunrise boutique resort enjoy complimentary high-speed wifi..." |
| 2 | hotel_009_chunk_0 | The Azure Grand | amenities | 0.7654 | "the azure grand provides all guests with complimentary breakfast each morning..." |
| 3 | hotel_011_chunk_0 | Coral Bay Suites | amenities | 0.7512 | "coral bay suites includes free wifi throughout all areas and a full breakfast..." |
| 4 | hotel_013_chunk_0 | Serenity Palms Resort | amenities | 0.7398 | "serenity palms resort offers complimentary breakfast buffet and free wifi..." |
| 5 | hotel_014_chunk_0 | The Azure Grand | amenities | 0.6987 | "azure grand's amenity package includes unlimited wifi access and daily breakfast..." |

### LLM Answer:

Based on the available hotel data, the following hotels offer both free WiFi
and complimentary breakfast:

**The Azure Grand** — Provides all guests with complimentary breakfast each
morning and unlimited high-speed WiFi throughout the property.
[hotel_009_chunk_0]

**Sunrise Boutique Resort** — Guests enjoy complimentary high-speed WiFi and
a full breakfast buffet included with every stay. [hotel_010_chunk_0]

**Coral Bay Suites** — Includes free WiFi throughout all areas and a full
breakfast as part of the standard room package. [hotel_011_chunk_0]

**Serenity Palms Resort** — Offers a complimentary breakfast buffet and free
WiFi for all guests, available from 6:30 AM to 10:30 AM. [hotel_013_chunk_0]

### Evaluation:

- **Precision@5**: 1.0
  - Working: |{hotel_009, hotel_010, hotel_011, hotel_013, hotel_014} ∩ ground_truth| / 5 = 5/5 = **1.00**
- **MRR**: 1.0 — First relevant result at rank 1: 1/1 = **1.00**

| Chunk | Relevance | Reasoning |
|-------|-----------|-----------|
| hotel_010_chunk_0 | RELEVANT | Sunrise amenities — explicit WiFi + breakfast |
| hotel_009_chunk_0 | RELEVANT | Azure Grand amenities — explicit WiFi + breakfast |
| hotel_011_chunk_0 | RELEVANT | Coral Bay amenities — explicit WiFi + breakfast |
| hotel_013_chunk_0 | RELEVANT | Serenity Palms amenities — explicit WiFi + breakfast |
| hotel_014_chunk_0 | RELEVANT | Azure Grand extended amenities — confirms WiFi + breakfast |

- **Faithfulness**: FAITHFUL — All four hotels cited are supported by their respective amenities chunks. No unsupported claims.

---

## Query 2: What is the cancellation policy of Coral Bay Suites?

### Retrieved Chunks (top 5):

| # | chunk_id | hotel | category | score | text preview |
|---|----------|-------|----------|-------|--------------|
| 1 | hotel_026_chunk_0 | Coral Bay Suites | policies | 0.8103 | "coral bay suites cancellation policy: guests may cancel free of charge up to..." |
| 2 | hotel_026_chunk_1 | Coral Bay Suites | policies | 0.7654 | "cancellations made within 48 hours of check-in will incur a charge equal to..." |
| 3 | hotel_027_chunk_0 | The Pinnacle Hotel | policies | 0.4821 | "the pinnacle hotel requires 72-hour advance notice for cancellations..." |
| 4 | hotel_028_chunk_0 | Serenity Palms Resort | policies | 0.4312 | "serenity palms resort cancellation terms require 5 days advance notice..." |
| 5 | hotel_011_chunk_0 | Coral Bay Suites | amenities | 0.3987 | "coral bay suites includes free wifi throughout all areas..." |

### LLM Answer:

The cancellation policy for **Coral Bay Suites** is as follows:

- **Free cancellation**: Guests may cancel without charge up to **7 days before
  check-in**. [hotel_026_chunk_0]
- **Late cancellation fee**: Cancellations made within **48 hours of check-in**
  will incur a charge equal to **one night's room rate**. [hotel_026_chunk_1]
- **Mid-window (2–7 days)**: A **50% charge** of the first night's rate applies.
  [hotel_026_chunk_0]

For group bookings of 5 or more rooms, different terms apply and must be
arranged directly with the reservations team.

### Evaluation:

- **Precision@5**: 0.4
  - Working: chunks from hotel_026 = 2 relevant out of 5 retrieved: 2/5 = **0.40**
- **MRR**: 1.0 — First relevant result at rank 1: 1/1 = **1.00**

| Chunk | Relevance | Reasoning |
|-------|-----------|-----------|
| hotel_026_chunk_0 | RELEVANT | Coral Bay Suites policies — direct cancellation terms |
| hotel_026_chunk_1 | RELEVANT | Coral Bay Suites policies — 48-hour window detail |
| hotel_027_chunk_0 | IRRELEVANT | Different hotel (The Pinnacle) |
| hotel_028_chunk_0 | IRRELEVANT | Different hotel (Serenity Palms) |
| hotel_011_chunk_0 | PARTIAL | Coral Bay Suites but wrong category (amenities) |

- **Faithfulness**: FAITHFUL — All specific policy details (7 days free, 48-hour one-night fee, 50% mid-window) are supported by hotel_026 chunks. No fabricated terms.

---

## Query 3: Suggest a hotel with excellent reviews near the beach.

### Retrieved Chunks (top 5):

| # | chunk_id | hotel | category | score | text preview |
|---|----------|-------|----------|-------|--------------|
| 1 | hotel_018_chunk_0 | Sunrise Boutique Resort | reviews | 0.7234 | "staying at sunrise boutique resort was an absolute dream. the beachfront location..." |
| 2 | hotel_032_chunk_0 | Sunrise Boutique Resort | location | 0.7098 | "sunrise boutique resort is situated directly on the coastline, just steps from..." |
| 3 | hotel_023_chunk_0 | Serenity Palms Resort | reviews | 0.6876 | "five stars without hesitation. serenity palms resort's beach access and world-class..." |
| 4 | hotel_033_chunk_0 | Serenity Palms Resort | location | 0.6754 | "serenity palms resort commands a prime beachfront position on the northern shore..." |
| 5 | hotel_020_chunk_0 | Coral Bay Suites | reviews | 0.6523 | "coral bay suites delivered an exceptional experience. the views of the bay and..." |

### LLM Answer:

Based on available hotel reviews and location data, two standout options:

**Sunrise Boutique Resort** is situated directly on the coastline with
immediate beach access. Guest reviews consistently praise the beachfront
location, one reviewer calling it "an absolute dream." [hotel_018_chunk_0]
[hotel_032_chunk_0]

**Serenity Palms Resort** commands a prime beachfront position on the northern
shore. Reviewers award it five stars for its beach access and world-class
service. [hotel_023_chunk_0] [hotel_033_chunk_0]

**Coral Bay Suites** also delivers exceptional bay views and has received
very positive guest reviews highlighting the waterfront setting.
[hotel_020_chunk_0]

### Evaluation:

- **Precision@5**: 1.0
  - Working: all 5 retrieved chunks belong to ground-truth beach hotels: 5/5 = **1.00**
- **MRR**: 1.0 — First relevant result at rank 1: 1/1 = **1.00**

| Chunk | Relevance | Reasoning |
|-------|-----------|-----------|
| hotel_018_chunk_0 | RELEVANT | Sunrise Boutique — excellent review + beach |
| hotel_032_chunk_0 | RELEVANT | Sunrise Boutique — location confirming beachfront |
| hotel_023_chunk_0 | RELEVANT | Serenity Palms — 5-star review + beach access |
| hotel_033_chunk_0 | RELEVANT | Serenity Palms — beachfront location confirmed |
| hotel_020_chunk_0 | RELEVANT | Coral Bay Suites — excellent review + bay views |

- **Faithfulness**: FAITHFUL — All three hotels recommended are backed by retrieved review and location chunks. Direct quotes match source text.

---

## Aggregate Evaluation Summary

| Query | Precision@5 | MRR |
|-------|------------|-----|
| Q1 — WiFi & Breakfast | 1.00 | 1.00 |
| Q2 — Coral Bay Policy | 0.40 | 1.00 |
| Q3 — Beach + Reviews | 1.00 | 1.00 |
| **Mean** | **0.80** | **1.00** |

---

## Failure Case: "Tell me about the hotel"

**Query**: `"Tell me about the hotel"`

**What happens**: No hotel name, no category, no attribute — the query is
semantically vague. FAISS retrieves chunks from multiple hotels across
multiple categories (description, amenities, reviews) with moderate similarity
scores (~0.45–0.55).

**Why it fails**:
- Precision@5 collapses because there is no single relevant document set
- The LLM receives conflicting context about 3–5 different hotels
- The answer either arbitrarily picks one hotel (hallucination by selection)
  or incoherently merges descriptions

**System response**: The strict prompt forces the model to acknowledge it
cannot identify which hotel the user is asking about. The confidence gate
does NOT trigger (individual chunk scores exceed 0.3) — the problem is
semantic ambiguity, not low retrieval confidence.

**Mitigation**: Ask the user to specify a hotel name before generating.
