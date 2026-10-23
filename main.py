"""
SHOOTLAB - AI Photo Evaluation microservice
--------------------------------------------
Deliberately RULE-BASED (no custom-trained ML model), per project scope:
we compute classic image-quality heuristics with PIL/NumPy instead of
calling a paid vision API or training a network. This keeps the module
feasible within the thesis timeline while still giving the user useful,
explainable feedback.

Metrics computed:
  - exposure    : how close the average brightness is to a well-exposed midtone
  - contrast    : standard deviation of luminance
  - sharpness   : variance of a Laplacian edge filter (focus estimate)
  - composition : how close the visually "busiest" region sits to a
                   rule-of-thirds intersection
  - noise       : local variance in low-detail regions (proxy for ISO noise)

Run:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="SHOOTLAB AI Evaluation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to the backend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DIMENSION = 1024  # downscale large photos before analysis for speed


def load_grayscale(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ.") from exc

    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    return gray


def score_exposure(gray: np.ndarray) -> tuple[float, str]:
    mean_brightness = float(gray.mean())  # 0-255
    # Well-exposed photos generally average ~90-160 for varied scenes.
    ideal_low, ideal_high = 90, 160
    if ideal_low <= mean_brightness <= ideal_high:
        score = 100.0
    elif mean_brightness < ideal_low:
        score = max(0.0, 100 - (ideal_low - mean_brightness) * 1.4)
    else:
        score = max(0.0, 100 - (mean_brightness - ideal_high) * 1.4)

    if mean_brightness < ideal_low - 25:
        note = "Ảnh hơi thiếu sáng (under-exposed) — thử mở khẩu độ lớn hơn, giảm tốc độ màn trập, hoặc tăng ISO."
    elif mean_brightness > ideal_high + 25:
        note = "Ảnh hơi dư sáng (over-exposed) — thử khép khẩu, tăng tốc độ màn trập, hoặc giảm ISO."
    else:
        note = "Phơi sáng cân đối."
    return round(score, 1), note


def score_contrast(gray: np.ndarray) -> tuple[float, str]:
    std = float(gray.std())  # typical range ~20-80
    ideal_low, ideal_high = 40, 75
    if ideal_low <= std <= ideal_high:
        score = 100.0
    elif std < ideal_low:
        score = max(0.0, (std / ideal_low) * 100)
    else:
        score = max(0.0, 100 - (std - ideal_high) * 1.2)

    if std < ideal_low - 10:
        note = "Ảnh khá bệt (contrast thấp) — cân nhắc điều chỉnh ánh sáng hoặc hậu kỳ tăng tương phản."
    else:
        note = "Độ tương phản ổn."
    return round(score, 1), note


def laplacian_variance(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    # Simple valid-mode 2D convolution via stride tricks (image sizes here are small enough).
    padded = np.pad(gray, 1, mode="edge")
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i : i + h, j : j + w]
    return float(out.var())


def score_sharpness(gray: np.ndarray) -> tuple[float, str]:
    var = laplacian_variance(gray)
    # Empirically, in-focus photos downsized to ~1024px land well above ~150-300 variance;
    # soft/blurry photos land much lower. This threshold is a heuristic, not a hard science.
    score = min(100.0, (var / 400.0) * 100)
    if score < 40:
        note = "Ảnh có dấu hiệu out-nét/rung tay — thử tăng tốc độ màn trập hoặc dùng chân máy."
    else:
        note = "Độ nét ổn."
    return round(score, 1), note


def score_composition(gray: np.ndarray) -> tuple[float, str]:
    h, w = gray.shape
    # Edge energy per pixel as a cheap "visual interest" proxy for where the subject sits.
    gy, gx = np.gradient(gray)
    energy = gx**2 + gy**2

    # Sum energy inside each cell of a 3x3 grid.
    ys = [0, h // 3, 2 * h // 3, h]
    xs = [0, w // 3, 2 * w // 3, w]
    best_cell, best_energy = None, -1.0
    for r in range(3):
        for c in range(3):
            cell = energy[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            e = float(cell.sum())
            if e > best_energy:
                best_energy = e
                best_cell = (r, c)

    # The 4 rule-of-thirds intersections correspond to the 4 "off-center" cells:
    # (0,0)(0,2)(2,0)(2,2). The center cell (1,1) is the least rule-of-thirds-friendly.
    thirds_cells = {(0, 0), (0, 2), (2, 0), (2, 2)}
    if best_cell in thirds_cells:
        score = 95.0
        note = "Chủ thể được đặt gần giao điểm quy tắc 1/3 — bố cục tốt."
    elif best_cell == (1, 1):
        score = 55.0
        note = "Chủ thể đang ở giữa khung hình — thử lệch chủ thể theo quy tắc 1/3 để bố cục sinh động hơn."
    else:
        score = 75.0
        note = "Bố cục tương đối cân đối, có thể tinh chỉnh gần hơn tới đường 1/3."
    return round(score, 1), note


def score_noise(gray: np.ndarray) -> tuple[float, str]:
    h, w = gray.shape
    gy, gx = np.gradient(gray)
    edge_energy = gx**2 + gy**2
    # "Flat" patches = bottom 40% of edge energy -> good proxy for background/sky/skin
    # where sensor noise (from high ISO) is most visible.
    flat_mask = edge_energy < np.percentile(edge_energy, 40)
    if flat_mask.sum() < 50:
        local_var = float(gray.var()) * 0.05  # fallback for busy/textured images
    else:
        local_var = float(gray[flat_mask].var())

    # Lower variance in flat regions = cleaner image.
    score = max(0.0, 100 - local_var * 1.5)
    score = min(100.0, score)
    if score < 55:
        note = "Có thể xuất hiện nhiễu hạt (ISO cao) ở vùng phẳng — cân nhắc hạ ISO hoặc khử nhiễu khi hậu kỳ."
    else:
        note = "Mức nhiễu chấp nhận được."
    return round(score, 1), note


@app.get("/health")
def health():
    return {"status": "ok", "service": "shootlab-ai-service"}


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh.")

    image_bytes = await file.read()
    gray = load_grayscale(image_bytes)

    exposure_score, exposure_note = score_exposure(gray)
    contrast_score, contrast_note = score_contrast(gray)
    sharpness_score, sharpness_note = score_sharpness(gray)
    composition_score, composition_note = score_composition(gray)
    noise_score, noise_note = score_noise(gray)

    weights = {
        "exposure": 0.25,
        "contrast": 0.15,
        "sharpness": 0.25,
        "composition": 0.25,
        "noise": 0.10,
    }
    breakdown = {
        "exposure": exposure_score,
        "contrast": contrast_score,
        "sharpness": sharpness_score,
        "composition": composition_score,
        "noise": noise_score,
    }
    overall = sum(breakdown[k] * weights[k] for k in weights)

    feedback = " ".join(
        [exposure_note, contrast_note, sharpness_note, composition_note, noise_note]
    )

    return {
        "overall_score": round(overall, 1),
        "breakdown": breakdown,
        "feedback": feedback,
    }
