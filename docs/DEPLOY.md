# Deploying the robomodel web app

The app is a single Python process (FastAPI + uvicorn) that serves the page and builds
models in separate worker processes.

## Run

```bash
docker build -t robomodel .
docker run -d --restart unless-stopped -p 8000:8000 --name robomodel robomodel
```

Put a reverse proxy (Caddy, nginx, a cloud load balancer) in front for HTTPS. Example
`Caddyfile`:

```
models.example.com {
    request_body { max_size 400MB }
    reverse_proxy 127.0.0.1:8000
}
```

## Built-in limits (`src/robomodel/web.py`)

| Setting | Default | Why |
|---|---|---|
| `MAX_UPLOAD_MB` | 400 | Largest CAD accepted |
| `MAX_JOBS` | 2 | Models built at the same time; more requests get HTTP 503 |
| `BUILD_TIMEOUT_S` | 600 | A build that takes longer is stopped (HTTP 504) |
| `RETENTION_HOURS` | 2 | Uploads and results are deleted after this; users can delete at once |

Sizing: a 1.3-million-triangle hull takes ~25 s and ~1.2 GB RAM; a 6-million-triangle ROV
assembly ~20 s and ~3.5 GB; large STEP assemblies (BlueROV2 R4, 186 MB) ~2.5 min and ~6 GB.
Give the machine at least `MAX_JOBS` × 4 GB of RAM, or lower `MAX_JOBS`.

## Before a public launch

- **Rate limiting** per IP at the proxy (uploads are expensive).
- **HTTPS only**; the app sets `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`.
- **three.js** is loaded from the jsDelivr CDN; for a stricter Content-Security-Policy or
  offline use, vendor it into `src/robomodel/static/`.
- **Privacy text and terms of use**: CAD files can be confidential. State the retention time
  (2 h), that files are not shared, and the accuracy limits (damping is an estimate).
- **Monitoring**: `GET /api/health`; log 5xx responses.
- **Licences**: robomodel's own licence is still to be chosen. Dependencies: trimesh (MIT),
  numpy/scipy (BSD), shapely (BSD), FastAPI (MIT), three.js (MIT), cascadio / OpenCASCADE
  (LGPL-2.1: fine to use as a library; keep it a separate, replaceable package).
