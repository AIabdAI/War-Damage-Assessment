# Model distribution: API, VPS deployment, and Flutter integration

This directory holds the backend half of the on-device deployment design
(dissertation §3.6). The models themselves are produced by
`scripts/compress_models.py` and measured by `scripts/evaluate_compressed.py`.

```
train (A100)  ->  compress (PTQ)  ->  evaluate  ->  MLflow registry
                                                        |
                                          serve/build_registry.py
                                                        |
                                         serve/model_api.py  (VPS)
                                                        |
                                     Flutter app: version check + download
```

## 1. Build and run locally

```bash
python scripts/compress_models.py --all            # produces models_mobile/
python scripts/evaluate_compressed.py --all --splits val,test
python serve/build_registry.py --production int8_static
uvicorn serve.model_api:app --host 0.0.0.0 --port 8000
```

Smoke-test the contract:

```bash
curl localhost:8000/api/v1/version
curl localhost:8000/api/v1/bundle/latest
curl -OJ localhost:8000/api/v1/models/yolo26m_det11-int8_static/download
```

## 2. The mobile update protocol

The app stores `bundle_version` plus each model's `sha256`.

| step | request | outcome |
|---|---|---|
| 1. cheap check | `GET /api/v1/version` | compare `bundle_version`; identical → stop, no download |
| 2. fetch plan | `GET /api/v1/bundle/latest` | metadata for detector + classifier (version, sha256, size, preprocessing) |
| 3. conditional download | `GET /api/v1/models/{id}/download` with `If-None-Match: <stored sha256>` | `304` if the device already has that artefact; `200` + bytes otherwise |
| 4. verify | — | recompute SHA-256 locally, compare with `X-Model-SHA256` before installing |
| 5. install | — | atomic replace; keep the previous file until verification succeeds |

Every response carries `ETag`, `X-Model-Version` and `X-Model-SHA256`, so the
client never re-downloads an artefact it already holds and never installs one
that failed integrity verification.

## 3. Flutter integration sketch

```dart
// 1. Is there anything new?
final v = jsonDecode((await http.get(Uri.parse('$base/api/v1/version'))).body);
if (v['bundle_version'] == prefs.getString('bundle_version')) return;

// 2. What do we need?
final b = jsonDecode((await http.get(Uri.parse('$base/api/v1/bundle/latest'))).body);

// 3. Conditional download + integrity check
for (final m in [b['detector'], b['classifier']]) {
  final res = await http.get(Uri.parse('$base${m['download_url']}'),
      headers: {'If-None-Match': prefs.getString('sha_${m['model_id']}') ?? ''});
  if (res.statusCode == 304) continue;                 // already installed
  final digest = sha256.convert(res.bodyBytes).toString();
  if (digest != m['sha256']) throw Exception('integrity check failed');
  await File('${dir.path}/${m['model_id']}.onnx').writeAsBytes(res.bodyBytes);
  await prefs.setString('sha_${m['model_id']}', digest);
}
await prefs.setString('bundle_version', b['bundle_version']);
```

On-device inference (packages: `onnxruntime` / `flutter_onnxruntime`, or
`tflite_flutter` once the TFLite conversion lands):

```
capture (camera only, no gallery)  ->  record GPS
  ->  preprocess (Table 3.5: CLAHE / sharpen / gamma / denoise)
  ->  detector.onnx        input 1x3x640x640, RGB, /255, letterboxed
  ->  crop each box (+10 % padding, as in training)
  ->  classifier.onnx      input 1x3x224x224, RGB, /255, ImageNet norm
  ->  store locally (SQLite)  ->  sync when online
```

The exact preprocessing contract per model is served in the registry entry
(`preprocessing` field), so the app never hard-codes these values.

### Detector output formats differ - check `output_format` in the registry

| model | ONNX output | on-device work |
|---|---|---|
| **yolo26m_det11** (recommended) | `(1, 300, 6)` = `[x1, y1, x2, y2, conf, class]` | **NMS is baked into the graph** - filter by confidence and use the boxes directly |
| yolo12s_det11 (compact option) | `(1, 15, 8400)` raw anchors (4 box + 11 class scores) | NMS must be implemented in Dart before use |

This is a concrete deployment advantage of the YOLO26 architecture: the
end-to-end export removes a whole class of on-device post-processing code
(and the bugs that come with it).

## 4. VPS deployment

```bash
# app user + code
adduser --system --group wda && mkdir -p /srv/wda && chown wda:wda /srv/wda
git clone https://github.com/AIabdAI/War-Damage-Assessment.git /srv/wda/app
cd /srv/wda/app && python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt fastapi uvicorn[standard]
.venv/bin/dvc pull models_mobile          # artefacts come from the DVC remote
.venv/bin/python serve/build_registry.py
```

`/etc/systemd/system/wda-model-api.service`:

```ini
[Unit]
Description=WDA model distribution API
After=network.target

[Service]
User=wda
WorkingDirectory=/srv/wda/app
Environment="MODEL_API_KEY=<generated-key>"
ExecStart=/srv/wda/app/.venv/bin/uvicorn serve.model_api:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

nginx (TLS terminates here; artefacts are static and immutable, so they cache
well):

```nginx
server {
    listen 443 ssl http2;
    server_name models.example.org;
    ssl_certificate     /etc/letsencrypt/live/models.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/models.example.org/privkey.pem;
    client_max_body_size 1m;
    location /api/ { proxy_pass http://127.0.0.1:8000; proxy_set_header Host $host; }
    location /health { proxy_pass http://127.0.0.1:8000; }
}
```

Also on the VPS (later stages of the platform):

| service | purpose |
|---|---|
| MLflow tracking server | central experiment + model registry (replaces the local sqlite files) |
| artefact store | S3-compatible bucket or a volume holding model binaries |
| sync API | assessment results uploaded from devices (GPS-tagged), feeding the data warehouse |
| retraining trigger | new user-corrected data → DVC → training pipeline → new model release |

Hardening checklist: HTTPS only, `MODEL_API_KEY` set, rate limiting on
`/api/v1/models/*/download`, read-only mount for `models_mobile/`, and no
write endpoints in this service (uploads belong to the separate sync API).

## 5. Releasing a new model version

1. Retrain or recompress → `scripts/compress_models.py`
2. Measure → `scripts/evaluate_compressed.py` (logs to MLflow)
3. Bump `MODEL_RELEASE` in `serve/build_registry.py`
4. `python serve/build_registry.py --production <variant>`
5. `dvc add models_mobile && dvc push && git commit && git push`
6. Restart the API service; devices detect the new `bundle_version` on their
   next check and download only what changed.
