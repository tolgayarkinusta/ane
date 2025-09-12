# SAHI / Tiled Çıkarım Entegrasyonu (ykn2.py)

Küçük ve hızlı hareket eden hedeflerde (kare, daire, üçgen) **tek kare üzerinde** çıkarım, nesne başına düşen piksel sayısı az olduğu için tespit performansını düşürebilir. Bu README, mevcut `ykn2.py` hattınıza **dilimli (tiled) çıkarım** eklemek için iki yol sunar:

1. **Ek bağımlılık yok**: *Yerel tiled inference* — SAHI fikrinin sade/yerel hâli (önerilen).  
2. **Tam SAHI entegrasyonu**: `ultralytics` + `sahi` ile hazır dilimleme API’si.

> **Özet:** Güçlü GPU’nuz var → `tile_size` ve `overlap` ile küçük nesnelerin piksellerini büyüterek **recall** artışı hedefliyoruz. FPS’i kontrol etmek için her kare yerine **her N karede** tiled çalıştırabilirsiniz.

---

## İçindekiler
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [1) Yerel Tiled Inference (0 bağımlılık)](#1-yerel-tiled-inference-0-bağımlılık)
  - [A. Yeni ayarlar](#a-yeni-ayarlar)
  - [B. `_process_yolo_output` güncellemesi](#b-_process_yolo_output-güncellemesi)
  - [C. `_detect_on_bgr` yardımcı fonksiyonu](#c-_detect_on_bgr-yardımcı-fonksiyonu)
  - [D. Sınıf-bazlı NMS](#d-sınıf-bazlı-nms)
  - [E. `_process_yolo_detection_tiled`](#e-_process_yolo_detection_tiled)
  - [F. `process_yolo_detection` entegrasyonu](#f-process_yolo_detection-entegrasyonu)
  - [Başlangıç ayarları & FPS stratejisi](#başlangıç-ayarları--fps-stratejisi)
- [2) Tam SAHI Entegrasyonu (opsiyonel)](#2-tam-sahi-entegrasyonu-opsiyonel)
  - [Seçenek A — Ultralytics `.pt` ile](#seçenek-a--ultralytics-pt-ile)
  - [Seçenek B — Mevcut hattı Sahi modeline sarmalamak](#seçenek-b--mevcut-hattı-sahi-modeline-sarmalamak)
  - [Video işleme & CLI](#video-işleme--cli)
- [Parametre Tuning Rehberi](#parametre-tuning-rehberi)
- [Küçük ve Hareketli Nesneler için Notlar](#küçük-ve-hareketli-nesneler-için-notlar)
- [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)
- [Benchmarking / A-B Test Önerisi](#benchmarking--a-b-test-önerisi)
- [Ek: Tracker ile kullanım](#ek-tracker-ile-kullanım)
- [Ek: Komut Özetleri](#ek-komut-özetleri)

---

## Hızlı Başlangıç

- **Model:** `yolo11l` veya `yolo11x` (küçük nesnelerde daha iyi tavan)
- **Tiled:** `tile_size=512`, `tile_overlap=0.20`, `tiled_conf_threshold=0.15`, `tiled_iou_threshold=0.45`
- **FPS kontrolü:** `tiling_every_n_frames=2` (her iki karede bir tiled)
- **IoU/Conf (NMS):** Objeler birbirine uzak → `iou≈0.55–0.65`, `conf≈0.20` başlangıç

---

## 1) Yerel Tiled Inference (0 bağımlılık)

Bu yöntem, mevcut `ykn2.py` hattınıza **ek kütüphane kurmadan** dilimleme ekler: görüntüyü overlap’li karolara böl, her karoda mevcut YOLO (TRT/ONNX) çıkarımı çalıştır, **tile ofsetleri** ile geri birleştir ve **sınıf-bazlı NMS** uygulayın.

### A. Yeni ayarlar
`HavaSavunmaArayuz.__init__` içine (diğer ayarların yanına):

```python
# --- Tiled (dilimli) çıkarım ayarları ---
self.use_tiled_inference = True       # Kapamak için False yapın
self.tile_size = 512                  # 512 / 640 iyi başlangıç
self.tile_overlap = 0.20              # %15–%25 arası deneyin
self.tiled_conf_threshold = 0.15      # Tiled içinde daha düşük konf
self.tiled_iou_threshold = 0.45       # Tiled birleştime NMS IoU
self.tiling_every_n_frames = 1        # 1: her kare, 2: iki karede bir ...
self._frame_index_for_tiling = 0
```

### B. `_process_yolo_output` güncellemesi
Eşik değerini **parametreli** kullanacak şekilde güncelleyin (geri uyumlu). Eski gövdeyi bununla değiştirin:

```python
def _process_yolo_output(self, output, img_width, img_height, classes_list, conf_threshold=None):
    """YOLO çıktısını işler ve sınırlayıcı kutuları döndürür."""
    boxes, confidences, class_ids = [], [], []
    thresh = CONF_THRESHOLD if conf_threshold is None else float(conf_threshold)

    predictions = np.squeeze(output).T
    scores = np.max(predictions[:, 4:], axis=1)
    valid = scores > thresh
    predictions = predictions[valid]
    scores = scores[valid]

    for i in range(len(predictions)):
        row = predictions[i]
        score = float(scores[i])
        class_id = int(np.argmax(row[4:]))
        cx, cy, w, h = row[:4]
        x = int((cx - w/2) * img_width  / IMG_WIDTH)
        y = int((cy - h/2) * img_height / IMG_HEIGHT)
        w = int(w * img_width  / IMG_WIDTH)
        h = int(h * img_height / IMG_HEIGHT)
        boxes.append([x, y, w, h])
        confidences.append(score)
        class_ids.append(class_id)

    if not boxes:
        return [], [], []
    idx = cv2.dnn.NMSBoxes(boxes, confidences, thresh, NMS_THRESHOLD)
    if len(idx) > 0:
        idx = idx.flatten().tolist()
        boxes = [boxes[i] for i in idx]
        confidences = [confidences[i] for i in idx]
        class_ids = [class_ids[i] for i in idx]
    return boxes, confidences, class_ids
```

### C. `_detect_on_bgr` yardımcı fonksiyonu
Mevcut TRT/ONNX yolunu **tek fonksiyonda** toplayıp bir BGR görüntü üzerinde kutuları döndüreceğiz:

```python
def _detect_on_bgr(self, bgr_img, model, classes_list, conf_threshold=None):
    inp = self._preprocess_frame_for_yolo(bgr_img)
    if inp is None:
        return [], [], []
    try:
        is_trt = (getattr(self, "model_is_tensorrt", False) and isinstance(model, str) and model == "tensorrt")
        if is_trt:
            if trt_context is None or trt_input_binding_idx is None or trt_output_binding_idx is None:
                return [], [], []
            np.copyto(trt_inputs[0]['host'], inp.flatten())
            cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], trt_stream)
            trt_context.execute_v2(trt_bindings)
            cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], trt_stream)
            trt_stream.synchronize()
            out = trt_outputs[0]['host'].reshape(
                trt_engine.get_tensor_shape(trt_engine.get_tensor_name(trt_output_binding_idx)))
        else:
            out = model.run([output_name], {input_name: inp})[0]
    except Exception:
        return [], [], []

    h, w = bgr_img.shape[:2]
    return self._process_yolo_output(out, w, h, classes_list, conf_threshold=conf_threshold)
```

> Not: Yukarıdaki parça mevcut `process_yolo_detection` fonksiyonunuzdaki TRT/ONNX ayrımını yeniden kullanır.

### D. Sınıf-bazlı NMS
Dilimlerden gelen tüm kutuları **sınıf bazında** birleştirmek için:

```python
def _nms_per_class(self, boxes, scores, class_ids, score_thresh, iou_thresh):
    if not boxes:
        return [], [], []
    keep_boxes, keep_scores, keep_cls = [], [], []
    boxes = np.array(boxes).tolist()
    scores = np.array(scores).astype(float).tolist()
    class_ids = np.array(class_ids).astype(int).tolist()
    for cid in sorted(set(class_ids)):
        idxs = [i for i,c in enumerate(class_ids) if c == cid]
        cls_boxes  = [boxes[i]  for i in idxs]
        cls_scores = [scores[i] for i in idxs]
        if not cls_boxes:
            continue
        keep_idx = cv2.dnn.NMSBoxes(cls_boxes, cls_scores, float(score_thresh), float(iou_thresh))
        if len(keep_idx) > 0:
            keep_idx = [idxs[i] for i in keep_idx.flatten().tolist()]
            for i in keep_idx:
                keep_boxes.append(boxes[i])
                keep_scores.append(scores[i])
                keep_cls.append(class_ids[i])
    return keep_boxes, keep_scores, keep_cls
```

### E. `_process_yolo_detection_tiled`
Dilimleme + çıkarım + birleştirme işleminin tamamı:

```python
def _process_yolo_detection_tiled(self, frame, model, classes_list):
    H, W = frame.shape[:2]
    ts = int(self.tile_size)
    step = max(1, int(ts * (1.0 - float(self.tile_overlap))))  # Örn. 512, %20 overlap → ~409
    all_boxes, all_scores, all_cls = [], [], []

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + ts, H)
            x1 = min(x0 + ts, W)
            tile = frame[y0:y1, x0:x1]
            bxs, scrs, cls = self._detect_on_bgr(
                tile, model, classes_list, conf_threshold=self.tiled_conf_threshold
            )
            for (x, y, w, h), s, c in zip(bxs, scrs, cls):
                all_boxes.append([x + x0, y + y0, w, h])
                all_scores.append(s)
                all_cls.append(c)

    f_boxes, f_scores, f_cls = self._nms_per_class(
        all_boxes, all_scores, all_cls,
        score_thresh=self.tiled_conf_threshold,
        iou_thresh=self.tiled_iou_threshold
    )

    detections = []
    for (x, y, w, h), s, c in zip(f_boxes, f_scores, f_cls):
        detections.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "score": float(s),
            "class_id": int(c),
            "class_name": classes_list[int(c)]
        })
    return detections
```

### F. `process_yolo_detection` entegrasyonu
Fonksiyonun başına küçük bir dal ekleyin:

```python
def process_yolo_detection(self, frame, model, classes_list):
    if getattr(self, "use_tiled_inference", False):
        self._frame_index_for_tiling = getattr(self, "_frame_index_for_tiling", 0) + 1
        setattr(self, "_frame_index_for_tiling", self._frame_index_for_tiling)
        if self._frame_index_for_tiling % int(getattr(self, "tiling_every_n_frames", 1)) == 0:
            return self._process_yolo_detection_tiled(frame, model, classes_list)
    # ↓ Mevcut tek-kare çıkarım yolunuz burada aynen devam etsin
    # input hazırlama → TRT/ONNX run → outputs_np → self._process_yolo_output(...)
```

#### Başlangıç ayarları & FPS stratejisi
- `tile_size = 512`, `tile_overlap = 0.20`, `tiled_conf_threshold = 0.15`, `tiled_iou_threshold = 0.45`
- FPS fazla düşerse: `tiling_every_n_frames = 2` veya `tile_size` ↑, `overlap` ↓
- Çok küçük hedef: `tile_size = 384`, `overlap = 0.25`
- Orta hedef: `tile_size = 640`, `overlap = 0.15`

---

## 2) Tam SAHI Entegrasyonu (opsiyonel)

### Seçenek A — Ultralytics `.pt` ile

```bash
pip install -U ultralytics sahi shapely
```

```python
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction

model = Yolov8DetectionModel(
    model_path="best.pt",            # .pt ağırlık
    confidence_threshold=0.15,
    device="cuda:0"
)
result = get_sliced_prediction(
    image=frame[..., ::-1],           # BGR→RGB
    detection_model=model,
    slice_height=512, slice_width=512,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.45
)
# result.object_prediction_list → kutular; arayüzünüzün beklediği dict’e çevirin
```

> Not: Bu yol `.onnx`/`.engine` yerine `.pt` ister. Mevcut borunuz TRT/ONNX ise Seçenek B’yi düşünün.

### Seçenek B — Mevcut hattı SAHI modeline sarmalamak
`from sahi.models.base import DetectionModel` türeterek `perform_inference` ve `postprocess_predictions` metodlarında yukarıdaki `_detect_on_bgr` mantığını kullanabilirsiniz. Böylece `get_sliced_prediction` doğrudan TRT/ONNX borunuzu dilimler.

### Video işleme & CLI

**Frame-by-frame:**
```python
import cv2, os
from sahi.predict import get_sliced_prediction

os.makedirs("out_frames", exist_ok=True)
cap = cv2.VideoCapture("video.mp4")
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = get_sliced_prediction(
        image=frame, detection_model=model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.6
    )
    res.export_visuals(export_dir="out_frames", file_name=f"{i:06d}.png", draw_labels=True)
    i += 1
cap.release()
# ffmpeg -r 30 -i out_frames/%06d.png -c:v libx264 -pix_fmt yuv420p video_sahi.mp4
```

**CLI (klasördeki görseller):**
```bash
sahi predict --source images/ \
  --model_type ultralytics --model_path best.pt --device cuda:0 \
  --slice_height 640 --slice_width 640 \
  --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 \
  --export_visual
```

---

## Parametre Tuning Rehberi

- **`tile_size`**: Küçük nesnelerde daha **küçük tile** (384–512) genelde daha iyi; çok küçültürsen tile sayısı ↑, FPS ↓.  
- **`tile_overlap`**: 0.15–0.25 arası güvenli aralık. Sınırdaki nesneleri kaçırmamak için **artır**, hız için **azalt**.  
- **`tiled_conf_threshold`**: 0.10–0.25 arası dene; düşük tutmak **recall**’ı artırır, birleştirme NMS duplikeleri temizler.  
- **`tiled_iou_threshold`**: 0.45 iyi başlangıç. Duplikeler kalırsa **düşür** (daha agresif NMS); nesneler siliniyorsa **yükselt**.  
- **Global NMS `iou`** (tek kare): Objeler aralıklı → orta–düşük aralık (0.55–0.65) çoğu zaman yeterli.  
- **`imgsz`**: 960/1280’e çıkarmak **kutunun piksel boyunu** artırır; maliyet kare ölçekli artar.

---

## Küçük ve Hareketli Nesneler için Notlar

- **Motion blur augment** ekleyin (Albumentations: `MotionBlur`, `Blur`, `RandomBrightnessContrast`).  
- Eğitimde **negatif** kareler (hedefsiz) ve farklı hız/ışık arka plan senaryoları ekleyin.  
- Gerçek çekimde mümkünse **yüksek enstantane** kullanın; aşırı blur en iyi modeli bile zorlar.

---

## Sık Karşılaşılan Sorunlar

- **Duplikeler artıyor:** `tiled_iou_threshold` ↓ (ör. 0.40–0.45) veya `tiled_conf_threshold` ↑.  
- **Nesneler kayboluyor:** `tiled_conf_threshold` ↓ (0.12–0.15), `tile_overlap` ↑, `tile_size` ↓ veya `imgsz` ↑.  
- **FPS düşük:** `tiling_every_n_frames` ↑, `tile_size` ↑, `tile_overlap` ↓, TRT yolunu kullanın.  
- **Görselleştirme yok:** SAHI’de `export_visuals(..., draw_labels=True)` kullanın; yerelde kendi çizim fonksiyonunuz çalışır.

---

## Benchmarking / A-B Test Önerisi

1) **A**: Tam-kare çıkarım (mevcut) vs **B**: Tiled (önerilen) için aynı veri üzerinde `precision/recall`, `FPS`, `GPU util/VRAM` ölçün.  
2) `tile_size` ∈ {384, 512, 640}, `overlap` ∈ {0.15, 0.20, 0.25} ızgarasında en iyi noktayı seçin.  
3) Video akışında **her N kare** tiled stratejisini de ölçün (N=1/2/3).

---

## Ek: Tracker ile kullanım

Tiled ile **daha düşük `conf`** + **yüksek recall** elde edip, video tarafında **ByteTrack / BoT-SORT** ile ID’leri koruyabilirsiniz.

```bash
yolo track model=best.pt source=video.mp4 conf=0.2 iou=0.65 tracker=bytetrack.yaml
# tracker parametrelerinde match_thresh (0.7→0.8) ile oynayabilirsiniz
```

> Not: NMS IoU (`iou`) ile takipteki `match_thresh` farklı kavramlardır; karıştırmayın.

---

## Ek: Komut Özetleri

**Kurulum (SAHI yolu):**
```bash
pip install -U ultralytics sahi shapely
```

**SAHI tek görsel:**
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",
    device="cuda:0",
    confidence_threshold=0.20,
    image_size=1280
)
res = get_sliced_prediction(
    image="frame.jpg",
    detection_model=detection_model,
    slice_height=640, slice_width=640,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.6
)
res.export_visuals(export_dir="runs/sahi", file_name="frame_sahi.png", draw_labels=True)
```

---

**Hazır.** Bu dosyadaki adımları takip ederek `ykn2.py` hattınıza tiled çıkarımı entegre edebilir, küçük ve hareketli hedeflerde kaçan tespitleri belirgin şekilde azaltabilirsiniz. Sorun durumda bu README’yi referans alarak parametreleri hızlıca ayarlayın; isterseniz sizin veri örneklerinizle **ayar matrisi** (sweep) de hazırlayabilirim.

