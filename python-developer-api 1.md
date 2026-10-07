# Plate Detection API — Integration Guide

This endpoint receives number-plate detection events (image + recognized
number) from the Python recognition service and stores them centrally.

## Endpoint

```
POST http://<server-host>:5000/api/edge/detections
```

## Authentication

Same shared secret used by the other edge endpoints (`EDGE_API_KEY`). Send it
as a Bearer token:

```
Authorization: Bearer <EDGE_API_KEY>
```

A missing or wrong key returns `401`.

## Request

`Content-Type: application/json`, body up to 10MB.

| Field      | Type   | Required | Notes |
|------------|--------|----------|-------|
| `cameraId` | string | yes      | The camera's id or `uniqueId` — either works. |
| `number`   | string | yes      | Recognized plate number. |
| `img`      | string | yes      | **Base64-encoded** image bytes (no `data:` URI prefix — raw base64 only). |
| `format`   | string | no       | `jpg` (default) or `png`. |

`updatedTime` is **not** a request field — the server stamps it itself the
moment the detection is stored, so it always reflects when the record was
received, not a value the client can set.

### Example (Python)

```python
import base64
import requests

with open("plate.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:5000/api/edge/detections",
    headers={"Authorization": "Bearer <EDGE_API_KEY>"},
    json={
        "cameraId": "8e4151a3-5a61-4640-ae93-adcc9b948523",
        "number": "GJ01AB1234",
        "img": img_b64,
        "format": "jpg",
    },
)
resp.raise_for_status()
print(resp.json())
```

### Example (curl)

```bash
curl -X POST http://localhost:5000/api/edge/detections \
  -H "Authorization: Bearer <EDGE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"cameraId":"<camera-id>","number":"GJ01AB1234","img":"<base64>"}'
```

## Response

**Success — `201`**
```json
{
  "success": true,
  "message": "Detection stored successfully",
  "data": {
    "id": "ddca2dbd-5b72-4f79-8d23-ed1f1c29da84",
    "cameraId": "8e4151a3-5a61-4640-ae93-adcc9b948523",
    "number": "GJ01AB1234",
    "updatedTime": "2026-09-02T07:18:56.424Z"
  }
}
```

**Errors**

| Status | Cause | Body |
|--------|-------|------|
| 400 | Missing `cameraId`/`number`/`img`, or `format` isn't `jpg`/`png` | `{"success":false,"message":"..."}` |
| 401 | Missing/invalid `Authorization` header | `{"success":false,"message":"Invalid or missing edge API key."}` |
| 404 | `cameraId` doesn't match any camera | `{"success":false,"message":"No camera found with id \"...\"."}` |
| 413 | Body over 10MB | `{"success":false,"message":"request entity too large"}` |

## Notes

- This route is exempt from the general API rate limit, so bursts of
  detections won't get throttled — it's still gated by the API key.
- Images are stored on disk under `storage/detections/<cameraId>/`, not
  returned as a public URL — only the DB record (id, number, updatedTime) is
  handed back in the response.
