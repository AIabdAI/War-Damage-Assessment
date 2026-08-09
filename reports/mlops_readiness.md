# تقرير جاهزية المشروع لإدخال أدوات MLOps (CML / MLflow / Docker)

- **تاريخ الإعداد:** 2026-08-08
- **المستودع:** `AIabdAI/War-Damage-Assessment` — الفرع `master`
- **الغرض:** حصر الوضع الحالي بدقة وتحديد الفجوات التي يجب سدّها قبل (وأثناء) إدخال CML وMLflow وDocker.

---

## 1. الوضع الحالي — ملخص تنفيذي

المشروع أنهى مرحلة جمع البيانات والتوصيف (dataset v1.0 موثّق عبر DVC) ونجح في تدريب دخاني (smoke training) يثبت أن السلسلة كاملة تعمل: بيانات → تحضير → تدريب YOLO → مقاييس. لكن البنية الحالية **غير جاهزة بعد** لأتمتة MLOps لثلاثة أسباب رئيسية:

1. **لا توجد pipeline معرّفة** (`dvc.yaml` غير موجود) — كل الخطوات تُشغَّل يدوياً بسكربتات منفصلة.
2. **لا يوجد CI فعّال** — ملف `data-guard.yml` مكتوب لكنه في جذر المستودع وليس في `.github/workflows/`، أي أنه **لا يعمل إطلاقاً**.
3. **كمية كبيرة من الشغل غير مُكوممت** — أدوات المراجعة والتدريب والإعدادات كلها untracked في git.

---

## 2. جرد تفصيلي

### 2.1 Git والكود

| البند | الحالة |
|---|---|
| الملفات المتتبعة في git | **15 ملفاً فقط** |
| ملفات غير متتبعة (untracked) | `class_check_tool.py`, `review_tool.py`, `verify_labels.py`, `train_smoke.py`, `prepare_smoke_dataset.py`, `configs/`, `data-guard.yml`, تقارير عدة |
| ملفات معدّلة غير مكوممتة | `annotation_tool.py`, `requirements.txt`, `processed_log.json`, `reports/annotation_stats.yaml` |
| فروع | `master` + `feature/*` + 6 فروع `report/annotation-*` قديمة تصلح للأرشفة |
| أكبر ملف في git | `processed_log.json` (1.27MB في git، **4.2MB حالياً على القرص وينمو**) — لا مكان له في git |

ملفات أوزان النماذج ملقاة في الجذر خارج أي تتبع: `yolo12n.pt`، `yolo26n.pt`، `yolov8s-worldv2.pt` (26MB)، `yolov8l-worldv2.pt` (94MB)، إضافة إلى `weights/clip/ViT-B-32.pt`. مجلدا `backup/` (631MB) و`.dvc/cache_backup/` (2.1GB) مخلفات تنظيف سابقة — **~2.7GB قابلة للاسترداد بعد التحقق** (كاش DVC الفعلي 1.4GB).

### 2.2 DVC والبيانات

- **الإصدار:** DVC 3.67.1 داخل `.venv` (Python 3.12.3). تنبيه معروف: dvc 2.x على النظام لا يقرأ ملفات المستودع — استخدم `.venv` دائماً.
- **Remote:** Google Drive (`gdrive://1Ts4w...`).
- **المتتبع عبر DVC:**
  - `data/raw` — 6,440 ملفاً / 635MB
  - `data/annotations` — 6,782 ملفاً / 462KB
- **غير موجود:** `dvc.yaml` (لا مراحل)، لا `params` مربوطة، لا `dvc metrics/plots`. مجلد `data/processed` ناتج محلي قابل لإعادة التوليد (قرار صحيح سابق بعدم تتبعه).

### 2.3 جودة البيانات (من `verification_report.md` بتاريخ 2026-08-08)

- 8,852 ملف label ممسوح، 13,426 صندوقاً، 12 كلاساً.
- **74 ملفاً فيها أخطاء صلبة:** class_id 13/16 خارج المدى (بقايا نظام الكلاسات القديم في ملفات `tile_*`)، وقيم w/h > 1.0 في ملفات `bricks_*` و`tile_*`.
- 4 labels يتيمة بلا صور، و14 صورة (`bricks_*(1)`) بلا label.
- مراجعة الكلاسات: أداة YOLO-World رصدت **403 ملفات مشتبهاً بها** (أبرزها Door/Wall_Cabinet)، ولم يُراجَع منها يدوياً سوى 6 صناديق من أصل 11,242.

### 2.4 التدريب

- تدريب دخاني ناجح على subset نظيف (10 epochs):
  - `yolo12n`: mAP50 ≈ **0.601**، mAP50-95 ≈ 0.382
  - `yolo26n`: mAP50 ≈ **0.624**، mAP50-95 ≈ 0.397
- البيئة المحلية جاهزة للتدريب: ultralytics 8.4.115، torch 2.5.1+cu124.
- التدريب الكامل مخطط له على A100 (المرحلة 6) — هذا يحدد شكل CML لاحقاً (runner خارجي).
- نتائج التدريب في `runs/` و`runs_smoke/` **غير متتبعة بأي أداة** — هذه بالضبط وظيفة MLflow القادمة.

### 2.5 الإعدادات (params) — تعارضات يجب حلها

- `configs/params.yaml` يقول `num_classes: 20`، بينما كل الأدوات والداتاسيت تعمل بـ **12 كلاساً**.
- `prepare_smoke_dataset.py` يبحث عن `params.yaml` في **جذر المشروع**، بينما الملف موجود في `configs/` — أي أن السكربت يسقط دائماً إلى القائمة المدمجة hard-coded.
- مسارات `params.yaml` (`data/raw/images`, `data/raw/labels`) لا تطابق البنية الفعلية (`data/raw` مسطّح + `data/annotations/labels`).

### 2.6 Docker — موجود شكلياً فقط

- `Dockerfile` بدائي: `python:3.11-slim` + تثبيت requirements + `COPY . .`
  - **لا يوجد `.dockerignore`** — أي build سينسخ `.venv` (غيغابايتات) و`data/` والأوزان داخل الصورة.
  - بايثون 3.11 في الصورة مقابل 3.12 في بيئة التطوير — عدم تطابق.
  - لا دعم GPU (لا صورة CUDA)، ولا مراحل multi-stage.
- `docker-compose.yml` أدنى حد (خدمة واحدة tty).
- `requirements.txt` **متضارب داخلياً**: `pandas` مرتان (مطلق ثم `==3.0.2`)، `Pillow` مرتان، `opencv-python` و`opencv-python-headless` معاً، و`ultralytics` معلّق رغم أنه مستخدم فعلياً. `pip install -r` الحالي غير قابل لإعادة الإنتاج.

### 2.7 CI / CML

- **لا يوجد مجلد `.github/` أصلاً.** ملف `data-guard.yml` (حارس منع البيانات في git) جاهز ومكتوب جيداً لكنه غير مفعّل لأنه في الجذر.
- لا CML، لا أي workflow.
- `tests/` **فارغ تماماً** — لا يوجد ما يشغّله CI حتى لو وُجد.

### 2.8 MLflow

- غير مثبّت وغير موجود في requirements. لا يوجد أي تتبع تجارب حالياً.

---

## 3. الفجوات مرتبة حسب الأولوية

| # | الفجوة | الخطورة | لماذا تعيق MLOps |
|---|---|---|---|
| 1 | `data-guard.yml` غير مفعّل (خارج `.github/workflows/`) | عالية | الحماية من تكرار حادثتي فقدان البيانات معطّلة، وهي أساس أي CI |
| 2 | شغل غير مكوممت (7+ أدوات وconfigs) | عالية | لا يمكن بناء CI/Docker على كود غير موجود في git |
| 3 | لا `dvc.yaml` pipeline | عالية | CML وMLflow يبنيان فوق pipeline قابلة لإعادة التشغيل (`dvc repro`) |
| 4 | `requirements.txt` متضارب | عالية | Docker build وCI سيفشلان أو يعطيان بيئة غير قابلة للتكرار |
| 5 | تعارض params (20 مقابل 12 كلاساً + مسار خاطئ) | متوسطة | مصدر الحقيقة الوحيد شرط لربط `dvc params` |
| 6 | أوزان النماذج (~130MB) و`processed_log.json` (4.2MB) في أماكن خاطئة | متوسطة | تفسد حجم git والصور الدوكرية |
| 7 | لا `.dockerignore` وDockerfile بدائي | متوسطة | صور ضخمة وغير آمنة (قد تسرّب بيانات) |
| 8 | `tests/` فارغ | متوسطة | CI بلا اختبارات = ضوء أخضر دائم بلا معنى |
| 9 | 74 ملف label بأخطاء + 403 ملفاً مشتبهاً بكلاساته | متوسطة | يجب حلها قبل وسم data-v1.1 والتدريب الكامل |
| 10 | فروع `report/*` قديمة ومجلدات backup | منخفضة | نظافة فقط |

---

## 4. خارطة الطريق المقترحة لإدخال الأدوات

### المرحلة A — ترتيب البيت (git) — نصف يوم
1. تنظيف `requirements.txt` (حسم التكرارات، تثبيت الإصدارات، تفعيل ultralytics أو فصله في `requirements-train.txt`).
2. توحيد params: ملف واحد (`params.yaml` في الجذر)، `num_classes: 12`، مسارات صحيحة، وتعديل السكربتات لقراءته.
3. نقل `data-guard.yml` إلى `.github/workflows/` — يصبح أول CI فعّال.
4. إضافة الأوزان إلى `.gitignore` (أو تتبعها بـ DVC في `weights/`)، وإخراج `processed_log.json` من git (`git rm --cached` + gitignore).
5. كوممت كل الأدوات غير المتتبعة، وحذف `backup/` و`.dvc/cache_backup/` بعد التحقق.

### المرحلة B — DVC pipeline — يوم
إنشاء `dvc.yaml` بأربع مراحل تغلّف السكربتات الموجودة أصلاً:
```
validate  (verify_labels.py)      → reports/validation/
prepare   (prepare_smoke_dataset) → data/processed/...
train     (train_smoke.py)        → weights + metrics.json
evaluate                          → dvc metrics/plots
```
مع ربط `params.yaml` كـ deps. بعدها `dvc repro` يعيد كل شيء، و`dvc metrics diff` يعطي أساس تقارير CML.

### المرحلة C — Docker — نصف يوم إلى يوم
1. `.dockerignore` (يستثني `.venv`, `data/`, `runs*`, `*.pt`, `.git`, `.dvc/cache`).
2. Dockerfile محدّث: Python 3.12، multi-stage، وتثبيت من requirements مثبّتة.
3. صورة ثانية اختيارية للتدريب مبنية على `pytorch/pytorch` مع CUDA (لسيناريو A100).
4. تحديث compose بخدمتين: `dev` و`train`.

### المرحلة D — CI + CML — يوم
1. workflow اختبارات: pytest على دوال التحقق (parse_clean_label وأمثالها) — يتطلب كتابة أول اختبارات في `tests/`.
2. workflow CML على PR: `dvc repro` على subset دخاني → `dvc metrics diff` → تعليق تلقائي بالمقاييس والرسوم على الـ PR.
3. للتدريب الكامل على A100: إما `cml runner` self-hosted أو تشغيل يدوي مع دفع المقاييس — يُحسم لاحقاً.
4. ملاحظة remote: مصادقة gdrive في CI مزعجة (OAuth تفاعلي) — تحتاج service account أو التفكير بنقل الـ remote (مثلاً S3/DagsHub) قبل أتمتة `dvc pull` في CI.

### المرحلة E — MLflow — نصف يوم
1. إضافة `mlflow` إلى requirements.
2. ultralytics يدعم MLflow مدمجاً: `yolo settings mlflow=True` + متغير `MLFLOW_TRACKING_URI` — يلتقط params/metrics/artifacts تلقائياً لكل run.
3. البداية بـ file-store محلي (`mlruns/` + gitignore)، ثم خادم tracking مشترك عند الحاجة (يمكن ضمه إلى docker-compose).
4. تسجيل الـ smoke runs الحالية كأساس مقارنة (yolo12n: mAP50=0.601، yolo26n: mAP50=0.624).

### بالتوازي — دين البيانات (لا يحجب الأدوات لكنه يحجب data-v1.1)
- إصلاح 74 ملفاً بأخطاء صلبة (class 13/16 وw/h>1) — أغلبها قابل للإصلاح آلياً (clamp + إعادة تعيين كلاس أو حذف).
- حسم 14 صورة بلا labels و4 labels يتيمة.
- استكمال مراجعة الـ 403 ملفات المشتبه بها عبر `review_tool.py`.

---

## 5. نقاط قوة تُبنى عليها

- انضباط ممتاز في فصل البيانات عن git (حوادث سابقة أنتجت `data-guard` وقواعد صارمة).
- سكربتات مكتوبة أصلاً بعقلية pipeline (كل سكربت CLI مستقل بمدخلات/مخرجات واضحة) — تغليفها في `dvc.yaml` شبه مباشر.
- بيئة تدريب محلية جاهزة (CUDA + ultralytics) وتدريب دخاني مثبت النجاح.
- توثيق جيد (تقارير تحقق ومراجعة كلاسات مؤتمتة).
