# ============================================================
# main.py — FastAPI Backend
# Sistem Klasifikasi Penyakit Kulit Eczema vs Psoriasis
# Berbasis MobileNetV2 + Grad-CAM XAI
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
import cv2
import base64
import io
import sys
import os
import json

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from gradcam   import GradCAM
from reasoning import RuleBasedReasoning, ood_detection

# ── Konfigurasi ──────────────────────────────────────────────
MODEL_PATH     = os.path.join(
    os.path.dirname(__file__), "models", "fase2_best.keras"
)
EMBEDDING_PATH = os.path.join(
    os.path.dirname(__file__), "models", "embedding_reference.json"
)
TARGET_LAYER   = "Conv_1"
CLASS_NAMES    = ["eczema", "psoriasis"]

# Threshold berdasarkan distribusi P5 training set:
# Eczema P5=0.6163, Psoriasis P5=0.5825
THRESHOLDS = {
    "eczema"   : 0.62,
    "psoriasis": 0.60,
}

# ── Inisialisasi FastAPI ─────────────────────────────────────
app = FastAPI(
    title       = "Skin XAI API",
    description = "API klasifikasi Eczema dan Psoriasis dengan Grad-CAM XAI",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:5173",
        "https://eczema-psoriasis-detection.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load reference embeddings ────────────────────────────────
embedding_ref = None
try:
    with open(EMBEDDING_PATH, "r") as f:
        embedding_ref = json.load(f)
    print("✅ Reference embeddings berhasil di-load.")
    for cls in CLASS_NAMES:
        print(f"   {cls}: {embedding_ref[cls]['n_samples']} samples, "
              f"sim_mean={embedding_ref[cls]['sim_mean']:.4f}, "
              f"sim_p5={embedding_ref[cls]['sim_p5']:.4f}")
except Exception as e:
    print(f"⚠️  Embedding reference tidak ditemukan: {e}")
    print("   Validasi embedding dinonaktifkan.")

# ── Load model dan komponen XAI ──────────────────────────────
print("Loading model...")
model      = None
base_model = None
gradcam    = None
reasoning  = None
gap_extractor = None

try:
    model      = tf.keras.models.load_model(MODEL_PATH)
    base_model = model.layers[1]
    gradcam    = GradCAM(model, base_model, TARGET_LAYER)
    reasoning  = RuleBasedReasoning()
    print("✅ Model dan komponen XAI berhasil di-load.")

    # Print semua layer untuk verifikasi
    print("   Struktur model:")
    for i, layer in enumerate(model.layers):
        print(f"   Layer {i}: {layer.name} ({type(layer).__name__})")

except Exception as e:
    print(f"❌ ERROR saat load model: {e}")

# ── Setup GAP extractor ──────────────────────────────────────
try:
    if model is not None:
        # Layer 2 = GlobalAveragePooling2D (sudah terkonfirmasi)
        gap_layer = model.layers[2]
        assert "global_average_pooling" in gap_layer.name.lower(), \
            f"Layer 2 bukan GAP: {gap_layer.name}"

        gap_extractor = tf.keras.Model(
            inputs  = model.inputs,
            outputs = gap_layer.output
        )
        print(f"✅ GAP extractor berhasil dibuat.")
        print(f"   Layer: {gap_layer.name}")
        print(f"   Output shape: {gap_extractor.output_shape}")
except Exception as e:
    print(f"⚠️  Gagal buat GAP extractor: {e}")
    gap_extractor = None


# ════════════════════════════════════════════════════════════
# FUNGSI VALIDASI INPUT
# ════════════════════════════════════════════════════════════

def validate_input_image(pil_img):
    """
    Validasi gambar sebelum inferensi menggunakan cosine similarity
    terhadap centroid feature vector training set.

    Threshold ditetapkan berdasarkan distribusi P5 training set:
    - Eczema    : 0.60 (P5 = 0.6163)
    - Psoriasis : 0.58 (P5 = 0.5825)

    Returns:
        dict: valid, reason, similarity, details
    """
    # Jika komponen tidak tersedia, loloskan semua gambar
    if gap_extractor is None or embedding_ref is None:
        print("⚠️  Validasi embedding dinonaktifkan (komponen tidak tersedia)")
        return {"valid": True, "reason": "", "similarity": 1.0, "details": {}}

    try:
        # Preprocess gambar
        img_resized = pil_img.resize((224, 224))
        img_array   = np.array(img_resized).astype("float32")

        # Validasi channel RGB
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return {
                "valid"     : False,
                "reason"    : (
                    "Format gambar tidak didukung. "
                    "Gunakan gambar berwarna (RGB/JPG/PNG)."
                ),
                "similarity": 0.0,
                "details"   : {}
            }

        # Normalisasi dan ekstrak feature
        img_norm    = tf.keras.applications.mobilenet_v2\
                        .preprocess_input(img_array)
        img_batch   = np.expand_dims(img_norm, axis=0)
        features    = gap_extractor.predict(img_batch, verbose=0)[0]

        # Hitung cosine similarity ke centroid setiap kelas
        similarities = {}
        for cls in CLASS_NAMES:
            centroid = np.array(embedding_ref[cls]["centroid"])
            sim = float(
                np.dot(features, centroid) /
                (np.linalg.norm(features) * np.linalg.norm(centroid) + 1e-8)
            )
            similarities[cls] = sim

        max_similarity = max(similarities.values())
        best_class     = max(similarities, key=similarities.get)
        best_threshold = THRESHOLDS[best_class]

        details = {
            "similarity_eczema"   : round(similarities["eczema"], 4),
            "similarity_psoriasis": round(similarities["psoriasis"], 4),
            "max_similarity"      : round(max_similarity, 4),
            "best_class"          : best_class,
            "threshold_used"      : best_threshold,
        }

        # Log untuk debugging
        print(
            f"Input validation: "
            f"sim_eczema={similarities['eczema']:.4f}, "
            f"sim_psoriasis={similarities['psoriasis']:.4f}, "
            f"max={max_similarity:.4f}, "
            f"threshold={best_threshold:.4f}, "
            f"valid={max_similarity >= best_threshold}"
        )

        if max_similarity < best_threshold:
            return {
                "valid"     : False,
                "reason"    : (
                    "Gambar yang diunggah tidak terdeteksi sebagai "
                    "gambar lesi kulit Eczema atau Psoriasis. "
                    "Sistem hanya dapat menganalisis gambar lesi "
                    "kulit yang termasuk kategori Eczema atau Psoriasis."
                ),
                "similarity": max_similarity,
                "details"   : details,
            }

        return {
            "valid"     : True,
            "reason"    : "",
            "similarity": max_similarity,
            "details"   : details,
        }

    except Exception as e:
        print(f"⚠️  Input validation error: {e}")
        return {"valid": True, "reason": "", "similarity": 1.0, "details": {}}


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def array_to_base64(img_array: np.ndarray) -> str:
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer  = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def heatmap_to_base64(heatmap: np.ndarray) -> str:
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return array_to_base64(heatmap_rgb)


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status"         : "ok",
        "message"        : "Skin XAI API berjalan",
        "model"          : "loaded" if model else "failed",
        "gap_extractor"  : "ready" if gap_extractor else "unavailable",
        "embedding_ref"  : "loaded" if embedding_ref else "unavailable",
    }


@app.get("/health")
def health_check():
    return {
        "status"        : "ok",
        "model_ready"   : model is not None,
        "gradcam"       : gradcam is not None,
        "reasoning"     : reasoning is not None,
        "gap_extractor" : gap_extractor is not None,
        "embedding_ref" : embedding_ref is not None,
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint utama: validasi input → inferensi → Grad-CAM → Reasoning.
    """
    if model is None or gradcam is None or reasoning is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum siap. Coba beberapa saat lagi."
        )

    if file.content_type not in [
        "image/jpeg", "image/png", "image/jpg", "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Gunakan JPG, PNG, atau WEBP."
        )

    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")

        # ── LAPISAN VALIDASI INPUT ────────────────────────────
        validation = validate_input_image(pil_img)

        if not validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "type"      : "input_validation_failed",
                    "message"   : validation["reason"],
                    "similarity": validation["similarity"],
                    "details"   : validation["details"],
                }
            )
        # ─────────────────────────────────────────────────────

        # Grad-CAM + inferensi
        gradcam_result = gradcam.explain(pil_img, show=False)

        # OOD Detection (post-hoc)
        ood = ood_detection(
            gradcam_result["confidence"],
            gradcam_result["all_probabilities"]
        )

        # Rule-Based Reasoning
        reasoning_result = reasoning.generate(
            predicted_class   = gradcam_result["predicted_class"],
            confidence        = gradcam_result["confidence"],
            heatmap           = gradcam_result["heatmap_resized"],
            all_probabilities = gradcam_result["all_probabilities"]
        )

        img_original_b64 = array_to_base64(gradcam_result["image_display"])
        img_heatmap_b64  = heatmap_to_base64(gradcam_result["heatmap_resized"])
        img_overlay_b64  = array_to_base64(gradcam_result["superimposed"])

        return JSONResponse(content={
            "status": "success",

            "classification": {
                "predicted_class"  : gradcam_result["predicted_class"],
                "nama_penyakit"    : reasoning_result["nama_penyakit"],
                "confidence"       : gradcam_result["confidence"],
                "confidence_label" : reasoning_result["confidence_label"],
                "all_probabilities": gradcam_result["all_probabilities"],
            },

            "reasoning": {
                "subtipe"          : reasoning_result["subtipe"],
                "subtipe_deskripsi": reasoning_result["subtipe_deskripsi"],
                "penjelasan_model" : reasoning_result["penjelasan_model"],
                "heatmap_interpret": reasoning_result["heatmap_interpret"],
                "fitur_klinis"     : reasoning_result["fitur_klinis"],
                "lokasi_predileksi": reasoning_result["lokasi_predileksi"],
                "rekomendasi"      : reasoning_result["rekomendasi"],
                "ambiguitas"       : reasoning_result["ambiguitas"],
                "ambiguitas_level" : reasoning_result["ambiguitas_level"],
            },

            "heatmap_analysis": {
                "quality"      : reasoning_result["heatmap_analysis"]["quality"],
                "coverage_pct" : reasoning_result["heatmap_analysis"]["coverage_pct"],
                "focus_score"  : reasoning_result["heatmap_analysis"]["focus_score"],
                "peak_location": reasoning_result["heatmap_analysis"]["peak_location"],
            },

            "ood": {
                "detected": ood["ood_detected"],
                "level"   : ood["level"],
                "pesan"   : ood["pesan"],
                "saran"   : ood["saran"],
            },

            "images": {
                "original": img_original_b64,
                "heatmap" : img_heatmap_b64,
                "overlay" : img_overlay_b64,
            }
        })

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat analisis: {str(e)}"
        )
        
@app.get("/ping")
def ping():
    return {"status": "ok"}