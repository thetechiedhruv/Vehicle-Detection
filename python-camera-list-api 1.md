# Camera List API — Integration Guide (for the Plate Detection service)

This endpoint returns the full list of registered cameras (with their
`id`) so the Python plate-recognition service knows which `cameraId` to
send when it later calls `POST /api/edge/detections`
(see `python-developer-api.md`).

## Endpoint

```
GET http://<server-host>:5000/api/edge/detections/cameras
```

## Authentication

None — same trust boundary as `POST /api/edge/detections`. No
`Authorization` header or `EDGE_API_KEY` is needed.

(This is different from `GET /api/edge/cameras`, which is used by the HLS
streamer and does require the `EDGE_API_KEY` Bearer token — the plate
detection service should use this endpoint instead.)

## Request

No query parameters or body. Returns every camera in the system
(not just online ones), each with only `id`, `name`, and `streamUrl`.

### Example (Python)

```python
import requests

resp = requests.get("http://localhost:5000/api/edge/detections/cameras")
resp.raise_for_status()
cameras = resp.json()["data"]["cameras"]

for cam in cameras:
    print(cam["id"], cam["name"], cam["streamUrl"])
```

### Example (curl)

```bash
curl http://localhost:5000/api/edge/detections/cameras
```

## Response

**Success — `200`**
```json
{
  "success": true,
  "message": "Cameras fetched successfully",
  "data": {
    "cameras": [
      {
        "id": "8e4151a3-5a61-4640-ae93-adcc9b948523",
        "name": "Sardar Patel Chowk",
        "streamUrl": "rtsp://192.168.1.107:554/stream1"
      }
    ]
  }
}
```

| Field       | Notes |
|-------------|-------|
| `id`        | Use this exact value as `cameraId` in `POST /api/edge/detections`. |
| `name`      | Human-readable camera name, for logging. |
| `streamUrl` | The configured RTSP/stream URL for this camera — use this to pull the feed. |

Only these three fields are returned. Camera login credentials
(`username`/`password`), `ipAddress`, and `status` are never included in
this response, even though the endpoint itself is unauthenticated.

## Notes

- This route is exempt from the general API rate limit, same as
  `POST /api/edge/detections`.
- The list is not paginated — it always returns every camera in one
  response.
