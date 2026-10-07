# PLATESCOPE EDGE

Live licence-plate recognition from RTSP cameras.

This is the camera version of `plate_scan.py`. Same detection, same OCR, same
voting — but instead of reading one video file from start to finish, it watches
entry/exit cameras continuously, records every vehicle, and syncs to a central
server.

If you have used the face/QR attendance service, this is the same thing for
number plates.

---

## Contents

1. [The two scripts, and when to use which](#1-the-two-scripts-and-when-to-use-which)
2. [How it works](#2-how-it-works)
3. [Install](#3-install)
4. [Configure](#4-configure)
5. [Run it](#5-run-it)
6. [Try it with no camera](#6-try-it-with-no-camera)
7. [Watch it work](#7-watch-it-work)
7b. [Seeing the camera live](#7b-seeing-the-camera-live-plate_watchpy---show)
8. [Read the results](#8-read-the-results)
9. [Every endpoint](#9-every-endpoint)
10. [Tuning](#10-tuning)
11. [Connecting a central server](#11-connecting-a-central-server)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. The two scripts, and when to use which

| | `plate_scan.py` | `plate_edge.py` |
|---|---|---|
| Input | one video file | live RTSP cameras |
| Runs | once, then exits | forever |
| Output | printed report + JSON | SQLite database + HTTP API |
| Answers | "what plates are in this video?" | "which vehicles came in today?" |

**Nothing was changed in `plate_scan.py`.** `plate_edge.py` imports its
functions, so both scripts share one copy of the detection and voting logic.

New files:

```
plate_edge.py     the live service
.env.example      settings template
README.md         this file
```

---

## 2. How it works

Each frame goes through two stages:

```
RTSP frame
    |
    v
[1] Vehicle detector (YOLO)  ->  finds cars/bikes/buses/trucks
    |                            and gives each a tracking ID
    v
    crop out each vehicle
    |
    v
[2] Plate detector (YOLO)    ->  finds the plate inside that crop
    |
    v
[3] PaddleOCR                ->  reads the characters
    |
    v
[4] Vote                     ->  several reads decide the final text
    |
    v
   log + send to server
```

Cropping the vehicle first is what makes distant plates readable — the plate
gets magnified before the plate detector ever sees it.

### The voting, and why it changed

`plate_scan.py` collects every read from the whole video, then decides at the
end. A camera has no end, so that cannot work here.

Instead, **each vehicle gets its own vote.** While a vehicle is on screen, its
tracking ID collects reads:

```
Vehicle #7 seen        ->  reads "GJ01SK360"    (1 vote)
Vehicle #7 seen again  ->  reads "GJ01SK3607"   (2 votes, grouped together)
Vehicle #7 seen again  ->  reads "GJ01SK3607"   (3 votes)
                              |
                              v
                       2 votes reached -> LOGGED, stop scanning this vehicle
```

Misreads like `GJ01SK360` and `GJ019SK3607` are folded into the same group
automatically, because OCR confuses `0/O`, `1/I`, `5/S` and so on.

A vehicle that drives off without ever reaching 2 votes is written to a
separate `unread_vehicle` table — so a camera reading nothing shows up as a
number, instead of silence.

### Registered vs unregistered

If a central server is configured, the service downloads a list of authorised
plates. Every detection is checked against it:

- **Registered** — logged with the owner name, sent to `/edge/vehicles/events`
- **Unregistered** — logged as unknown, sent to `/edge/vehicles/unregistered`

Without a central server, everything is logged locally as unregistered.

---

## 3. Install

### What you need

- Python 3.8+
- The model file `models/license-plate-finetune-v1x.pt`
- A camera that gives an RTSP URL (or just a video file — see section 6)

### Step 1 — virtual environment

```bash
cd vehical_detectoin

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

Your prompt should now start with `(venv)`.

### Step 2 — install the requirements

**If you do not have an NVIDIA GPU** (most laptops and small edge boxes), install
torch from the CPU index first. This skips ~3 GB of CUDA libraries and is by far
the biggest time saving:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**If you do have an NVIDIA GPU**, just:

```bash
pip install -r requirements.txt
```

Either way this takes a few minutes — torch and paddle are large. Everything
runs fine on CPU; the timings quoted in `plate_scan.py` are CPU numbers.

### Step 3 — check the model file exists

```bash
ls -lh models/license-plate-finetune-v1x.pt
```

If that file is missing the service will not start. The vehicle model
(`yolo12m.pt`) downloads by itself the first time if it is not there.

---

## 4. Configure

Copy the template:

```bash
cp .env.example .env
```

Now open `.env` and fill in your camera. **You only need these two lines to
start:**

```ini
ENTRY_CAMERA_ID=gate-1
ENTRY_CAMERA_URL=rtsp://admin:yourpassword@192.168.1.64:554/Streaming/Channels/102
```

Leave everything else as it is. Blank `CENTRAL_API_URL` means standalone mode —
it runs perfectly and just stores everything locally.

### Finding your RTSP URL

The format depends on the brand:

| Brand | URL |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |

Use the **substream** (`102`, or `subtype=1`), not the main stream. It is lower
resolution, far lighter on CPU, and still plenty for plates.

Test the URL before you configure it:

```bash
ffplay "rtsp://admin:yourpassword@192.168.1.64:554/Streaming/Channels/102"
```

If ffplay shows video, the service will connect too. If it does not, fix the
URL first — the problem is the camera, not this code.

### Adding an exit camera

```ini
EXIT_CAMERA_ID=gate-2
EXIT_CAMERA_URL=rtsp://admin:yourpassword@192.168.1.65:554/Streaming/Channels/102
```

Entry and exit run on separate threads with separate models, so they do not
slow each other down.

---

## 5. Run it

```bash
source venv/bin/activate
python plate_edge.py
```

You should see:

```
============================================================
  PLATESCOPE EDGE — Live Plate Recognition
============================================================
  Site:    Plate Edge Site
  Port:    8002
  Central: standalone
============================================================

INFO - Starting PlateScope Edge...
INFO - Standalone mode: skipping registered plate sync
INFO - Started camera gate-1 (entry)
INFO - [gate-1] loading models...
INFO - [gate-1] models ready
INFO - [gate-1] connecting...
INFO - [gate-1] connected
```

Loading models takes 10–30 seconds. **This is normal — wait for
`models ready`.**

When a vehicle is read:

```
INFO - [gate-1] UNREGISTERED GJ01SK3607 votes=3 motorcycle -> entry
```

Stop it with `Ctrl+C`.

---

## 6. Try it with no camera

You do not need a camera to test. `plate_edge.py` opens video files the same
way it opens streams, so you can feed it one of your test videos.

Start the service with no camera configured, then in a **second terminal**:

```bash
curl -X POST http://localhost:8002/cameras/start \
  -H "Content-Type: application/json" \
  -d '{
        "camera_id": "test",
        "rtsp_url": "videos/bike_plates.mp4",
        "purpose": "entry",
        "name": "test video"
      }'
```

The video plays through and plates get detected exactly as they would from a
camera. When it reaches the end the service treats it as a dropped connection
and replays it from the start, so it loops until you stop it:

```bash
curl -X POST http://localhost:8002/cameras/test/stop
```

This is the fastest way to confirm your install works before touching a camera.

---

## 7. Watch it work

Open in a browser:

```
http://localhost:8002/stream/gate-1
```

You get the live video with:

- a **box around each plate** with the current read
- **green text** — plate confirmed and logged
- **orange text** — still collecting votes
- a bottom line showing active vehicles and total events

Replace `gate-1` with whatever you set as `ENTRY_CAMERA_ID`.

> If you started a test video as above, the stream is at
> `http://localhost:8002/stream/test`.

---

## 7b. Seeing the camera live (`plate_watch.py --show`)

`plate_edge.py` streams to a browser. For debugging you usually want the raw
camera in a window with the detector's reasoning drawn on top:

```bash
./venv/bin/python plate_watch.py --show
```

Keys: **q** quit, **p** pause, **s** save a snapshot.

Over SSH there is no display, so write a file instead and scrub through it:

```bash
./venv/bin/python plate_watch.py --save-video check.mp4
```

### Reading the picture

The colours tell you which stage is failing, which is the whole point:

| Colour | Meaning | If you are stuck here |
|---|---|---|
| **Blue** box | vehicle found — `motorcycle #7  2/3` votes so far | — |
| **Green** box | plate read and accepted | working |
| **Red** box | plate read but **rejected** — shows the raw text and why | see the label |
| **Grey** box | plate located, OCR returned nothing | too blurry/small |
| **no boxes** | no vehicle detected at all | lower `VEHICLE_CONF`, raise `VEHICLE_IMGSZ` |

A red label looks like `GJ01SK <- too short (7, need 6)`. That is the single
most useful line when plates are being seen but never logged: OCR is working
and validation is throwing the read away.

The HUD in the corner shows resolution, frames, passes, **ms per pass** (if
this is larger than `DETECT_INTERVAL` × 1000 the machine cannot keep up),
vehicles tracked, plates logged, and a one-line hint naming the likely cause.

### Why this is not in plate_edge.py's browser stream

`plate_scan.plates_in_image()` returns text that has already passed through
`clean_plate_text()`, so a rejected read comes back as an empty string —
indistinguishable from "no plate there". `plate_watch.py` keeps the raw string
so it can show you the difference. Validation still calls
`plate_scan.clean_plate_text`, so what counts as a valid plate is identical to
the live service.

---

## 8. Read the results

### Last 50 detections

```bash
curl http://localhost:8002/events
```

```json
{
  "success": true,
  "data": {
    "events": [
      {
        "plate": "GJ01SK3607",
        "camera_id": "gate-1",
        "event_type": "entry",
        "vehicle_type": "motorcycle",
        "votes": 3,
        "confidence": 0.94,
        "registered": 0,
        "owner": null,
        "timestamp": "2026-09-02T12:31:07.412000+00:00",
        "synced": 0
      }
    ]
  }
}
```

More at once:

```bash
curl "http://localhost:8002/events?limit=200"
```

### Is everything healthy?

```bash
curl http://localhost:8002/status
```

Shows camera status, active vehicle tracks, event counts and pending syncs.

### Straight from the database

Everything is in `data/plates.db`:

```bash
sqlite3 data/plates.db "SELECT timestamp, plate, camera_id, event_type, votes
                        FROM plate_log ORDER BY id DESC LIMIT 20;"
```

Today's entries:

```bash
sqlite3 data/plates.db "SELECT plate, COUNT(*) FROM plate_log
                        WHERE event_type='entry' AND timestamp > date('now')
                        GROUP BY plate;"
```

Vehicles that could not be read:

```bash
sqlite3 data/plates.db "SELECT COUNT(*) FROM unread_vehicle;"
```

Export to CSV:

```bash
sqlite3 -header -csv data/plates.db "SELECT * FROM plate_log;" > plates.csv
```

**Tables:**

| Table | Holds |
|---|---|
| `plate_log` | every plate that was successfully read |
| `unread_vehicle` | vehicles seen but never read (missed plates) |
| `sync_queue` | events waiting to reach the central server |

They are kept separate on purpose, so `plate_log` stays a clean record of real
detections.

---

## 9. Every endpoint

Base URL: `http://localhost:8002`

| Method | Path | Does |
|---|---|---|
| GET | `/health` | quick alive check |
| GET | `/status` | full status: cameras, counts, settings |
| GET | `/cameras` | list cameras and their health |
| POST | `/cameras/start` | start a camera at runtime |
| POST | `/cameras/{id}/stop` | stop one camera |
| GET | `/stream/{id}` | live MJPEG video with boxes |
| GET | `/events?limit=N` | recent detections |
| GET | `/plates` | the registered plate list |
| POST | `/sync/plates` | re-download the registered list now |

Interactive docs — every endpoint with a **Try it out** button:

```
http://localhost:8002/docs
```

Starting a camera without restarting the service:

```bash
curl -X POST http://localhost:8002/cameras/start \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"gate-3","rtsp_url":"rtsp://...","purpose":"exit"}'
```

`purpose` must be `entry` or `exit`.

---

## 10. Tuning

All of these live in `.env`. Restart the service after changing them.

| Setting | Default | Raise it when | Lower it when |
|---|---|---|---|
| `MIN_VOTES` | `3` | wrong plates are getting logged | plates are being missed |
| `PLATE_IMGSZ` | `320` | plates are small/far (try `640`) | CPU cannot keep up |
| `VEHICLE_IMGSZ` | `480` | distant vehicles are missed | CPU cannot keep up |
| `DETECT_INTERVAL` | `0.3` | the machine is falling behind | vehicles pass too fast to reach MIN_VOTES |
| `PLATE_COOLDOWN` | `60` | the same vehicle logs twice | vehicles legitimately return quickly |
| `TRACK_TTL` | `3.0` | vehicles are briefly hidden behind others | — |
| `MATCH_MAX_DISTANCE` | `1` | registered plates are not matching | wrong vehicle is being authorised |

**The two that matter most:**

`MIN_VOTES` is the accuracy/coverage trade, and it matters more here than it
does in `plate_scan.py`. That script pools every read from the whole video and
decides at the end, so later reads correct earlier ones. A live camera cannot
wait — the vote closes while the vehicle is still in frame, and once it closes
that vehicle is not scanned again.

Measured on `videos/bike_plates.mp4` (9 bikes, ground truth known) using this
service's live per-vehicle voting:

| Setting | Result |
|---|---|
| `MIN_VOTES=2`, every 10 frames | 8 plates — 2 degraded, 1 missed |
| `MIN_VOTES=3`, every 6 frames | **9 plates, all correct** |

At `2` the vote closed on the first two agreeing reads, so the truncated
`BG0244` locked in before the full `GJ27BG0244` was ever read. Hence the
default of `3`.

`DETECT_INTERVAL` is your speed limit. One pass is two YOLO models plus OCR —
roughly 0.3–1.0s on CPU. Setting it below what your machine can actually do
does not gain you anything. If the log falls behind real time, raise it to
`0.6` or `0.8`, or drop `PLATE_IMGSZ`.

**Fast-moving traffic isn't reaching the vote threshold?** Lower
`DETECT_INTERVAL` first — more passes while the vehicle is in frame means more
reads to vote with. Dropping `MIN_VOTES` is the last resort: it does not find
more plates, it just accepts weaker evidence for them.

You can measure the trade on your own footage before committing to it:

```bash
python plate_watch.py --source your_clip.mp4 --min-votes 2
python plate_watch.py --source your_clip.mp4 --min-votes 3
```

---

## 11. Connecting a central server

Fill these in `.env`:

```ini
SITE_ID=your-site-id
EDGE_API_KEY=your-key
CENTRAL_API_URL=https://your-server.com/api/v1
```

The service then:

- downloads authorised plates at startup and every `SYNC_INTERVAL` seconds
- posts each detection as it happens
- sends a heartbeat with camera health every `HEARTBEAT_INTERVAL` seconds
- **queues everything to SQLite when the network is down, and retries every
  60 seconds** — no detections are lost while offline

> **Important:** the endpoint paths below are *assumptions*, extrapolated from
> the face-attendance service. Check them against your real API and correct
> them in `.env` — they are all configurable, no code change needed.
>
> ```ini
> EP_REGISTERED=/edge/vehicles/sync
> EP_EVENT=/edge/vehicles/events
> EP_UNREGISTERED=/edge/vehicles/unregistered
> EP_HEARTBEAT=/edge/heartbeat
> ```

Force a refresh of the plate list without restarting:

```bash
curl -X POST http://localhost:8002/sync/plates
```

---

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
The venv is not active, or step 2 was skipped.
`source venv/bin/activate && pip install -r requirements.txt`

**OCR returns nothing, or PaddleOCR raises an argument error**
You have paddleocr 2.x installed. The code needs 3.x:
`pip install -U "paddleocr>=3.0.0" "paddlepaddle>=3.0.0"`

**Vehicles are detected but never get a tracking ID**
`lap` is missing — ultralytics needs it for `.track()`. `pip install lap`

**`[gate-1] cannot open stream`**
The camera URL is wrong or unreachable. Test it with `ffplay` (section 4). Check
the password has no `@` in it — if it does, it breaks the URL and must be
percent-encoded as `%40`.

**Service starts but nothing is ever detected**
1. Open `http://localhost:8002/stream/gate-1` and look. Do you see green boxes
   around vehicles? If not, the vehicle detector is not finding them — raise
   `VEHICLE_IMGSZ` to `640`.
2. Vehicles boxed but no plate text? Raise `PLATE_IMGSZ` to `640`.
3. Check `sqlite3 data/plates.db "SELECT COUNT(*) FROM unread_vehicle;"` — a
   high number means vehicles are being seen but plates not read. Usually the
   camera angle: it needs a clear, near-straight-on view of the plate.

**Plates read but slightly wrong (`GJ01SK360` instead of `GJ01SK3607`)**
Raise `MIN_VOTES` to `3`. More reads means the correct spelling wins more often.

**Same vehicle logged twice**
Raise `PLATE_COOLDOWN` from `60` to `300`.

**It is running very slowly / falling behind**
In order: raise `DETECT_INTERVAL` to `0.8`, drop `PLATE_IMGSZ` to `256`, drop
`VEHICLE_IMGSZ` to `320`, switch `VEHICLE_MODEL_PATH` to `yolov8s.pt`. Also
confirm you are on the camera **substream**, not the main stream.

**Port 8002 already in use**
Change `PORT` in `.env`.

**Where are the logs?**
Printed to the terminal. To keep them:
`python plate_edge.py 2>&1 | tee edge.log`

---

## Quick reference

```bash
# setup, once
cd vehical_detectoin
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env          # then edit ENTRY_CAMERA_URL

# run
python plate_edge.py

# watch          http://localhost:8002/stream/gate-1
# api docs       http://localhost:8002/docs
# results        curl http://localhost:8002/events
```
