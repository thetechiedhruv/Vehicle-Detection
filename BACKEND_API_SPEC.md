> **SUPERSEDED.** The backend implements a single endpoint,
> `POST /api/edge/detections`, documented in `python-developer-api 1.md`.
> `plate_edge.py` now uses that. This file is kept only as a record of
> the richer payload the edge can produce (votes, owner, vehicle type,
> entry/exit) should the backend ever want it.

# Vehicle / ANPR Edge API — specification for the backend

**For:** Node backend developer
**Server:** `http://192.168.1.52:5000/api`
**Consumer:** `plate_edge.py`, the edge service running at each site

---

## Context

An edge service now runs at the site reading licence plates from RTSP cameras.
It works and is detecting plates today, but it cannot report them — none of the
routes below exist yet.

**Four routes are needed.** Until they exist the edge service stores everything
locally in SQLite and keeps it queued, so nothing is lost in the meantime; it
will drain the backlog once the routes are live.

The existing `GET /api/edge/cameras` already works and is the pattern to
follow — same auth, same response envelope.

### Current state, verified against the running server

| Route | Status |
|---|---|
| `GET /api/edge/cameras` | ✅ works |
| `GET /api/edge/vehicles/sync` | ❌ 404 — **build this** |
| `POST /api/edge/vehicles/events` | ❌ 404 — **build this** |
| `POST /api/edge/vehicles/unregistered` | ❌ 404 — **build this** |
| `POST /api/edge/heartbeat` | ❌ 404 — **build this** |

> **Paths are negotiable.** Every one is configurable on the edge side via
> `.env`. If your routing convention differs, tell us the real paths and we
> change a config line — no code change. What must not change is the **request
> and response shapes** below.

---

## Auth

Identical to `GET /edge/cameras`. Every request carries:

```http
Authorization: Bearer <EDGE_API_KEY>
X-Site-ID: <YOUR-SITE-ID>
Content-Type: application/json
```

Reject with `401` when the key is missing or wrong, as you do now.

## Response envelope

Same as the existing route:

```json
{ "success": true, "message": "...", "data": { } }
```

The edge service only checks `success`. **Anything other than HTTP 200 with
`success: true` is treated as a failure and the event is re-queued and retried
every 60 seconds** — so return 200 only when the record is genuinely stored.

One exception: a **404 is treated as "not implemented"** and the edge service
disables that endpoint for the run rather than retrying. Do not return 404 for
"vehicle not found" — use 200 with an empty result, or 400.

---

## 1. `GET /edge/vehicles/sync`

Returns the plates authorised for this site. The edge service calls this at
startup and every 5 minutes, and matches every detection against it to decide
registered vs unregistered.

### Request

```http
GET /api/edge/vehicles/sync?siteId=<YOUR-SITE-ID>
```

### Response

```json
{
  "success": true,
  "message": "Vehicles fetched successfully",
  "data": {
    "vehicles": [
      {
        "_id": "6964d4103836519a0ffbe999",
        "plateNumber": "GJ01SK3607",
        "ownerName": "Dharmik Panchani",
        "vehicleType": "motorcycle",
        "active": true
      }
    ]
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `_id` | string | yes | stored and echoed back as `vehicleId` on events |
| `plateNumber` | string | yes | see normalisation below |
| `ownerName` | string | no | shown in edge logs |
| `vehicleType` | string | no | free text: `car`, `motorcycle`, … |
| `active` | boolean | no | defaults to `true` |

### Plate normalisation — important

The edge service strips everything that is not `A–Z` or `0–9` and uppercases,
so `GJ-01 SK 3607` becomes `GJ01SK3607`. **Send plates the same way**, or at
least consistently — anything after cleaning that is under 6 or over 12
characters is dropped with a warning.

Matching tolerates one character of OCR confusion (`0/O`, `1/I`, `5/S`, `8/B`),
so a read of `GJ01SK36O7` still matches `GJ01SK3607`.

### Empty list

Return `200` with `"vehicles": []`. The edge service keeps its previous list
rather than wiping it, so an empty response never accidentally de-authorises
every vehicle.

---

## 2. `POST /edge/vehicles/events`

One **registered** vehicle was seen. Sent immediately on detection.

### Request body

```json
{
  "plate": "GJ01SK3607",
  "siteId": "<YOUR-SITE-ID>",
  "cameraId": "camera_1",
  "cameraName": "entry camera",
  "eventType": "entry",
  "vehicleType": "motorcycle",
  "trackId": 7,
  "votes": 3,
  "variants": ["GJ01SK360", "GJ019SK3607"],
  "confidence": 0.94,
  "registered": true,
  "vehicleId": "6964d4103836519a0ffbe999",
  "owner": "Dharmik Panchani",
  "timestamp": "2026-09-02T13:15:02.412000+00:00",
  "capturedImage": "/9j/4AAQSkZJRgABAQAA..."
}
```

| Field | Type | Notes |
|---|---|---|
| `plate` | string | cleaned, uppercase, no separators |
| `siteId` | string | Mongo ObjectId |
| `cameraId` | string | as configured on the edge |
| `cameraName` | string | human label |
| `eventType` | string | **`entry` or `exit` only** |
| `vehicleType` | string | from the detector: `car`, `motorcycle`, `bus`, `truck` |
| `trackId` | integer | per-camera tracking id; **not unique across cameras or restarts** — do not use as a key |
| `votes` | integer | independent reads that agreed. Higher = more confident |
| `variants` | string[] | the misreads that were folded in. Useful for debugging OCR |
| `confidence` | float | 0–1, OCR character confidence |
| `registered` | boolean | always `true` on this route |
| `vehicleId` | string | the `_id` from the sync list |
| `owner` | string\|null | |
| `timestamp` | string | **ISO 8601, UTC** — when detected on the edge, not when received |
| `capturedImage` | string\|null | base64 JPEG, no `data:` prefix. Vehicle crop, max 480px wide, ~20–80 KB |

### Response

```json
{ "success": true, "message": "Event recorded", "data": { "id": "..." } }
```

### Please make it idempotent

If the network drops mid-request the edge service re-queues and resends the
same event. Treat `siteId` + `cameraId` + `plate` + `timestamp` as a unique
key and return `200` on a duplicate rather than inserting twice.

---

## 3. `POST /edge/vehicles/unregistered`

Identical body, for a plate that is **not** in the sync list. `registered` is
`false`, and `vehicleId` and `owner` are `null`.

Kept as a separate route because these usually want different handling —
alerts, a review queue, an enrolment flow. If you would rather have one route,
say so and we will point both at it; the `registered` boolean already
distinguishes them.

---

## 4. `POST /edge/heartbeat`

Every 30 seconds. Tells you the site is alive and whether its cameras are
healthy.

### Request body

```json
{
  "siteId": "<YOUR-SITE-ID>",
  "siteName": "ais",
  "service": "plate_recognition",
  "timestamp": "2026-09-02T13:15:02.412000+00:00",
  "status": "online",
  "registeredPlates": 42,
  "cameraCount": 1,
  "onlineCameras": 1,
  "cameras": [
    {
      "cameraId": "camera_1",
      "name": "entry camera",
      "purpose": "entry",
      "status": "online",
      "isReachable": true,
      "activeTracks": 2,
      "eventsEmitted": 17,
      "lastFrameAgeMs": 120,
      "startedAt": "2026-09-02T12:56:16.517000+00:00",
      "lastChecked": "2026-09-02T13:15:02.412000+00:00"
    }
  ]
}
```

`status` per camera is one of `online`, `connecting`, `error`, `stopped`.
A site whose heartbeat stops arriving for a few minutes is down.

Note `service: "plate_recognition"` — the face service sends
`face_recognition` to the same route, so you can share one handler.

### Response

```json
{ "success": true, "message": "Heartbeat received" }
```

---

## Suggested storage

```js
// vehicles — the authorised list served by /edge/vehicles/sync
{ _id, siteId, plateNumber, ownerName, vehicleType, active, createdAt }

// vehicle_events — one row per detection
{
  _id, siteId, cameraId, cameraName,
  plate, eventType,            // entry | exit
  vehicleType, registered,
  vehicleId,                   // null when unregistered
  owner,
  votes, confidence, variants,
  imageUrl,                    // write capturedImage to disk/S3, store the path
  detectedAt,                  // from `timestamp` — the edge clock
  receivedAt                   // your server clock
}
```

Index `vehicle_events` on `{ siteId, detectedAt }` and on `{ plate, detectedAt }`.

**Do not store `capturedImage` as base64 in Mongo.** At ~50 KB each and a few
hundred events a day it will bloat the collection fast. Decode it, write the
JPEG to disk or object storage, and keep the path.

**Keep both timestamps.** Edge devices buffer while offline, so a batch of
events can arrive hours after `detectedAt`. Reporting must use `detectedAt`.

---

## How to test before the edge is pointed at you

```bash
KEY="<edge api key>"
SITE="<YOUR-SITE-ID>"

# 1. sync
curl "http://192.168.1.52:5000/api/edge/vehicles/sync?siteId=$SITE" \
  -H "Authorization: Bearer $KEY" -H "X-Site-ID: $SITE"

# 2. an event
curl -X POST "http://192.168.1.52:5000/api/edge/vehicles/events" \
  -H "Authorization: Bearer $KEY" -H "X-Site-ID: $SITE" \
  -H "Content-Type: application/json" \
  -d '{"plate":"GJ01SK3607","siteId":"'$SITE'","cameraId":"camera_1",
       "cameraName":"entry camera","eventType":"entry","vehicleType":"motorcycle",
       "trackId":7,"votes":3,"variants":[],"confidence":0.94,"registered":true,
       "vehicleId":null,"owner":null,
       "timestamp":"2026-09-02T13:15:02.412000+00:00","capturedImage":null}'

# 3. heartbeat
curl -X POST "http://192.168.1.52:5000/api/edge/heartbeat" \
  -H "Authorization: Bearer $KEY" -H "X-Site-ID: $SITE" \
  -H "Content-Type: application/json" \
  -d '{"siteId":"'$SITE'","siteName":"ais","service":"plate_recognition",
       "timestamp":"2026-09-02T13:15:02.412000+00:00","status":"online",
       "registeredPlates":0,"cameraCount":1,"onlineCameras":1,"cameras":[]}'
```

All three should return `200` with `success: true`.

---

## Checklist

- [ ] `GET  /edge/vehicles/sync` returns `data.vehicles[]`
- [ ] `POST /edge/vehicles/events` stores and returns `success: true`
- [ ] `POST /edge/vehicles/unregistered` same, for unknown plates
- [ ] `POST /edge/heartbeat` accepts and returns `success: true`
- [ ] All four require the Bearer key and return `401` without it
- [ ] Duplicate events are idempotent, not double-inserted
- [ ] `capturedImage` is written to storage, not into Mongo
- [ ] Non-200 only when the write genuinely failed (the edge retries on failure)

Questions on any field: ask, before building around a guess.
