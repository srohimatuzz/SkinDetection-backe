# ============================================================
# src/reasoning.py
# Diekstrak dari: 02_GradCAM.ipynb
# Berisi: class RuleBasedReasoning, fungsi ood_detection
# Digunakan oleh: app.py (Streamlit prototype)
# ============================================================

import numpy as np

# Konfigurasi global — harus konsisten dengan training
CLASS_NAMES = ["eczema", "psoriasis"]


# ============================================================
# FUNGSI: ood_detection
# Sumber: Cell G-14a (tambahan setelah G-14)
# Posisi: post-hoc guardrail, BUKAN kontribusi utama
# ============================================================

def ood_detection(confidence, all_probabilities):
    """
    Deteksi apakah gambar kemungkinan di luar distribusi training.

    Mekanisme: threshold berbasis confidence score dan gap probabilitas.
    Keterbatasan: tidak efektif untuk gambar OOD yang secara visual
    mirip dengan kelas training (terbukti gagal pada gambar panu
    dengan confidence 82.7%).

    Args:
        confidence        : float, nilai confidence kelas prediksi
        all_probabilities : dict {"eczema": float, "psoriasis": float}

    Returns:
        dict berisi ood_detected (bool), level, pesan, saran
    """
    ecz_prob = all_probabilities.get("eczema", 0.0)
    pso_prob = all_probabilities.get("psoriasis", 0.0)
    gap      = abs(ecz_prob - pso_prob)

    # Threshold diperketat: confidence < 0.75 sudah beri warning sedang
    if confidence < 0.60:
        if gap < 0.20:
            return {
                "ood_detected": True,
                "level"       : "TINGGI",
                "pesan"       : (
                    "Model tidak dapat mengklasifikasikan gambar ini "
                    "dengan keyakinan yang cukup. Kemungkinan gambar "
                    "menunjukkan kondisi kulit di luar kategori Eczema "
                    "dan Psoriasis, atau kualitas gambar tidak memadai."
                ),
                "saran": (
                    "Pastikan gambar menampilkan lesi kulit dengan jelas. "
                    "Sistem hanya valid untuk Eczema dan Psoriasis."
                )
            }
        else:
            return {
                "ood_detected": True,
                "level"       : "SEDANG",
                "pesan"       : (
                    "Confidence rendah. Gambar mungkin tidak termasuk "
                    "kategori Eczema atau Psoriasis yang dikenali sistem."
                ),
                "saran": "Konsultasikan dengan dokter untuk diagnosis akurat."
            }

    elif confidence < 0.75 and gap < 0.30:
        return {
            "ood_detected": True,
            "level"       : "RENDAH",
            "pesan"       : (
                "Sistem mendeteksi kemungkinan ambiguitas. "
                "Hasil prediksi perlu dikonfirmasi lebih lanjut."
            ),
            "saran": "Disarankan konsultasi dengan dokter spesialis kulit."
        }

    return {
        "ood_detected": False,
        "level"       : None,
        "pesan"       : "",
        "saran"       : ""
    }


# ============================================================
# CLASS: RuleBasedReasoning
# Sumber: Cell G-9 FINAL (versi dengan perbaikan interpretasi
#         heatmap kontekstual dan deteksi ambiguitas 3 level)
# Kontribusi penelitian — JANGAN MODIFIKASI LOGIKA KLINIS
# ============================================================

class RuleBasedReasoning:
    """
    Sistem penjelasan teks otomatis berbasis aturan klinis.

    Komponen:
    1. Deteksi subtipe klinis (Eczema: akut/kronik/hiperkeratotik,
       Psoriasis: plak/gutata/inversa)
    2. Kategorisasi confidence (4 tier)
    3. Interpretasi heatmap kontekstual per kelas
    4. Deteksi ambiguitas (3 level)
    """

    CLINICAL_FEATURES = {
        "eczema": {
            "nama_id": "Eksim (Eczema)",
            "subtipe": {
                "akut": {
                    "ciri"     : "eritema, vesikel, basah",
                    "deskripsi": (
                        "Fase akut ditandai eritema difus dengan batas "
                        "tidak tegas, dapat disertai vesikel dan eksudasi"
                    )
                },
                "kronik": {
                    "ciri"     : "likenifikasi, sisik kasar",
                    "deskripsi": (
                        "Fase kronik ditandai penebalan kulit (likenifikasi), "
                        "sisik kasar, dan dapat terbentuk retakan (fissure)"
                    )
                },
                "hiperkeratotik": {
                    "ciri"     : "plak tebal, retakan",
                    "deskripsi": (
                        "Tipe hiperkeratotik menunjukkan plak tebal "
                        "berskuama kasar terutama di telapak tangan dan kaki"
                    )
                }
            },
            "karakteristik_umum": [
                "pruritus (gatal) sebagai gejala utama",
                "perjalanan penyakit kronik-relaps",
                "kulit kering (xerosis) di area sekitar lesi",
                "dapat terjadi likenifikasi pada fase kronik",
                "sisik bervariasi: halus (akut) hingga kasar (kronik/hiperkeratotik)"
            ],
            "lokasi_umum": [
                "lipatan siku dan lutut (fossa antecubiti)",
                "pergelangan tangan dan punggung tangan",
                "telapak tangan dan kaki (tipe hiperkeratotik)",
                "leher dan area periorbital",
                "tungkai bawah"
            ],
            # Eczema: distribusi difus → heatmap menyebar VALID secara klinis
            "heatmap_menyebar_valid": True
        },

        "psoriasis": {
            "nama_id": "Psoriasis",
            "subtipe": {
                "plak": {
                    "ciri"     : "plak tebal, skuama putih",
                    "deskripsi": (
                        "Tipe plak (vulgaris) paling umum: plak eritematosa "
                        "tebal dengan skuama putih keperakan berlapis"
                    )
                },
                "gutata": {
                    "ciri"     : "papul kecil tersebar",
                    "deskripsi": (
                        "Tipe gutata: papul kecil seperti tetesan air "
                        "tersebar di badan, sering muncul setelah infeksi"
                    )
                },
                "inversa": {
                    "ciri"     : "area lipatan, merah halus",
                    "deskripsi": (
                        "Tipe inversa: eritema merah halus tanpa skuama "
                        "tebal di area lipatan kulit (aksila, inguinal)"
                    )
                }
            },
            "karakteristik_umum": [
                "plak eritematosa dengan skuama berlapis",
                "skuama berwarna putih keperakan",
                "batas lesi tegas dan terdefinisi jelas",
                "fenomena Auspitz (titik darah saat skuama dilepas)",
                "fenomena Koebner pada area trauma"
            ],
            "lokasi_umum": [
                "kulit kepala (scalp) dan garis rambut",
                "lutut dan siku (area ekstensor)",
                "punggung bawah dan sakrum",
                "kuku (onikolisis, pitting)",
                "telapak tangan dan kaki"
            ],
            # Psoriasis: seharusnya fokus pada plak → menyebar tidak ideal
            "heatmap_menyebar_valid": False
        }
    }

    CONFIDENCE_RULES = {
        "sangat_tinggi": {
            "range"      : (0.90, 1.00),
            "label"      : "Sangat Tinggi",
            "deskripsi"  : (
                "Model memiliki keyakinan sangat tinggi. Pola visual "
                "sangat konsisten dengan karakteristik kelas yang diprediksi."
            ),
            "rekomendasi": (
                "Hasil dapat dijadikan rujukan awal yang kuat "
                "untuk evaluasi klinis."
            )
        },
        "tinggi": {
            "range"      : (0.75, 0.90),
            "label"      : "Tinggi",
            "deskripsi"  : (
                "Model memiliki keyakinan tinggi. Sebagian besar fitur "
                "visual konsisten dengan kelas yang diprediksi."
            ),
            "rekomendasi": (
                "Hasil dapat dijadikan bahan pertimbangan untuk "
                "evaluasi klinis lebih lanjut."
            )
        },
        "sedang": {
            "range"      : (0.60, 0.75),
            "label"      : "Sedang",
            "deskripsi"  : (
                "Model memiliki keyakinan sedang. Terdapat ambiguitas "
                "visual antara kedua kelas."
            ),
            "rekomendasi": (
                "Disarankan evaluasi klinis langsung oleh "
                "dokter spesialis kulit."
            )
        },
        "rendah": {
            "range"      : (0.50, 0.60),
            "label"      : "Rendah",
            "deskripsi"  : (
                "Model memiliki keyakinan rendah. Pola visual "
                "tidak cukup diskriminatif."
            ),
            "rekomendasi": (
                "Hasil tidak dapat dijadikan acuan. Pemeriksaan "
                "klinis langsung sangat diperlukan."
            )
        }
    }

    # ----------------------------------------------------------
    # METODE INTERNAL
    # ----------------------------------------------------------

    def get_confidence_tier(self, confidence):
        """Kategorisasi confidence ke 4 tier."""
        for tier, info in self.CONFIDENCE_RULES.items():
            low, high = info["range"]
            if low <= confidence <= high:
                return tier, info
        return "rendah", self.CONFIDENCE_RULES["rendah"]

    def analyze_heatmap(self, heatmap):
        """
        Analisis kuantitatif heatmap Grad-CAM.

        Returns:
            dict berisi focus_score, coverage_pct, peak_location, quality
        """
        h           = heatmap / (heatmap.max() + 1e-8)
        coverage    = float((h > 0.5).mean() * 100)
        focus_score = float(h.std())

        # Lokasi peak aktivasi
        peak_y, peak_x = np.unravel_index(np.argmax(h), h.shape)
        h_h, h_w       = h.shape
        rel_y          = peak_y / h_h
        rel_x          = peak_x / h_w

        vert  = "atas"   if rel_y < 0.33 else \
                "tengah" if rel_y < 0.66 else "bawah"
        horiz = "kiri"   if rel_x < 0.33 else \
                "pusat"  if rel_x < 0.66 else "kanan"

        if focus_score > 0.35 and coverage < 30:
            quality = "fokus"
        elif focus_score > 0.25:
            quality = "sedang"
        else:
            quality = "menyebar"

        return {
            "focus_score"   : focus_score,
            "coverage_pct"  : coverage,
            "peak_location" : f"{vert}-{horiz}",
            "quality"       : quality
        }

    def interpret_heatmap_contextual(self, predicted_class, heatmap_info, confidence):
        """
        Interpretasi heatmap yang mempertimbangkan konteks kelas.

        Perbaikan utama dari Cell G-9 FINAL:
        Confidence tinggi + heatmap menyebar pada Eczema =
        distribusi lesi difus yang VALID secara klinis,
        bukan indikasi ketidakyakinan model.
        """
        quality        = heatmap_info["quality"]
        coverage       = heatmap_info["coverage_pct"]
        peak_loc       = heatmap_info["peak_location"]
        menyebar_valid = self.CLINICAL_FEATURES[predicted_class][
            "heatmap_menyebar_valid"
        ]

        # Kasus 1: Confidence tinggi + heatmap menyebar
        if confidence >= 0.75 and quality == "menyebar":
            if menyebar_valid:
                return (
                    f"Area aktivasi Grad-CAM menyebar di seluruh gambar "
                    f"(coverage: {coverage:.1f}%), konsisten dengan "
                    f"karakteristik lesi yang berdistribusi difus. Model "
                    f"menggunakan pola global gambar sebagai dasar keputusan "
                    f"dengan keyakinan tinggi."
                )
            else:
                return (
                    f"Area aktivasi Grad-CAM menyebar (coverage: {coverage:.1f}%). "
                    f"Meskipun confidence tinggi, aktivasi yang menyebar "
                    f"mengindikasikan model mempertimbangkan fitur global gambar. "
                    f"Evaluasi klinis tetap diperlukan."
                )

        # Kasus 2: Confidence rendah + heatmap menyebar
        elif confidence < 0.75 and quality == "menyebar":
            return (
                f"Area aktivasi Grad-CAM menyebar ke berbagai region "
                f"(coverage: {coverage:.1f}%), indikasi ambiguitas visual "
                f"antara kedua kondisi. Model tidak menemukan fitur "
                f"diskriminatif yang dominan."
            )

        # Kasus 3: Heatmap fokus (ideal)
        elif quality == "fokus":
            return (
                f"Area aktivasi Grad-CAM terpusat pada region {peak_loc} "
                f"gambar (coverage: {coverage:.1f}%), indikasi model "
                f"mendeteksi fitur lesi yang spesifik dan terlokalisir."
            )

        # Kasus 4: Heatmap sedang
        else:
            return (
                f"Area aktivasi Grad-CAM cukup terpusat di area {peak_loc} "
                f"(coverage: {coverage:.1f}%), model mendeteksi beberapa "
                f"fitur relevan pada region tersebut."
            )

    def infer_subtipe(self, predicted_class, heatmap_info, confidence):
        """
        Estimasi subtipe klinis berdasarkan karakteristik heatmap.
        Rule sederhana berbasis coverage dan focus_score.
        """
        subtipe_info = self.CLINICAL_FEATURES[predicted_class]["subtipe"]

        if predicted_class == "eczema":
            if heatmap_info["coverage_pct"] > 35:
                return "akut", subtipe_info["akut"]
            elif heatmap_info["coverage_pct"] < 20:
                return "hiperkeratotik", subtipe_info["hiperkeratotik"]
            else:
                return "kronik", subtipe_info["kronik"]
        else:  # psoriasis
            if heatmap_info["focus_score"] > 0.35:
                return "plak", subtipe_info["plak"]
            elif heatmap_info["coverage_pct"] > 40:
                return "gutata", subtipe_info["gutata"]
            else:
                return "plak", subtipe_info["plak"]

    # ----------------------------------------------------------
    # METODE UTAMA
    # ----------------------------------------------------------

    def generate(self, predicted_class, confidence,
                heatmap, all_probabilities=None):
        """
        Generate seluruh output Rule-Based Reasoning.

        Args:
            predicted_class   : str ("eczema" atau "psoriasis")
            confidence        : float
            heatmap           : np.array (heatmap_resized 224x224)
            all_probabilities : dict {"eczema": float, "psoriasis": float}

        Returns:
            dict lengkap berisi semua output reasoning
        """
        cls_info                = self.CLINICAL_FEATURES[predicted_class]
        tier, conf_info         = self.get_confidence_tier(confidence)
        heatmap_info            = self.analyze_heatmap(heatmap)
        subtipe_key, subtipe_info = self.infer_subtipe(
            predicted_class, heatmap_info, confidence
        )
        heatmap_interpret       = self.interpret_heatmap_contextual(
            predicted_class, heatmap_info, confidence
        )

        nama_penyakit = cls_info["nama_id"]

        # Penjelasan model lengkap
        penjelasan_model = (
            f"Model mengidentifikasi pola yang konsisten dengan "
            f"{nama_penyakit} (kemungkinan tipe {subtipe_key}). "
            f"{subtipe_info['deskripsi']}. {heatmap_interpret}"
        )

        # Informasi klinis
        fitur_teks = (
            f"Karakteristik umum {nama_penyakit}: "
            + "; ".join(cls_info["karakteristik_umum"][:3]) + "."
        )

        lokasi_teks = (
            f"Lokasi predileksi: "
            + ", ".join(cls_info["lokasi_umum"][:3]) + "."
        )

        # Deteksi ambiguitas 3 level
        other_class = [c for c in CLASS_NAMES if c != predicted_class][0]
        other_prob  = all_probabilities.get(other_class, 0.0) \
                    if all_probabilities else 0.0
        other_nama  = self.CLINICAL_FEATURES[other_class]["nama_id"]

        if other_prob > 0.35:
            ambiguitas       = (
                f"AMBIGUITAS TINGGI: Model mendeteksi kemungkinan "
                f"{other_nama} sebesar {other_prob:.1%}. Kemiripan visual "
                f"kedua kondisi sangat tinggi pada kasus ini, evaluasi "
                f"klinis langsung sangat diperlukan."
            )
            ambiguitas_level = "tinggi"
        elif other_prob > 0.25:
            ambiguitas       = (
                f"Ambiguitas sedang: Terdapat kemungkinan {other_nama} "
                f"({other_prob:.1%}). Disarankan konfirmasi klinis."
            )
            ambiguitas_level = "sedang"
        else:
            ambiguitas       = ""
            ambiguitas_level = "rendah"

        return {
            "predicted_class"   : predicted_class,
            "nama_penyakit"     : nama_penyakit,
            "confidence"        : confidence,
            "confidence_tier"   : tier,
            "confidence_label"  : conf_info["label"],
            "subtipe"           : subtipe_key,
            "subtipe_deskripsi" : subtipe_info["deskripsi"],
            "penjelasan_model"  : penjelasan_model,
            "heatmap_interpret" : heatmap_interpret,
            "fitur_klinis"      : fitur_teks,
            "lokasi_predileksi" : lokasi_teks,
            "conf_deskripsi"    : conf_info["deskripsi"],
            "rekomendasi"       : conf_info["rekomendasi"],
            "ambiguitas"        : ambiguitas,
            "ambiguitas_level"  : ambiguitas_level,
            "other_class_prob"  : other_prob,
            "heatmap_analysis"  : heatmap_info,
        }