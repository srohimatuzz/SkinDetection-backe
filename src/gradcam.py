# ============================================================
# src/gradcam.py
# Diekstrak dari: 02_GradCAM.ipynb
# Berisi: class GradCAM, class GuidedGradCAM
# Digunakan oleh: app.py (Streamlit prototype)
# ============================================================

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Konfigurasi global — harus konsisten dengan training
CLASS_NAMES = ["eczema", "psoriasis"]
IMG_SIZE    = (224, 224)


# ============================================================
# CLASS 1: GradCAM
# Sumber: Cell G-2 REVISED (versi final yang dipakai)
# Kontribusi penelitian utama — JANGAN MODIFIKASI LOGIKA
# ============================================================

class GradCAM:
    """
    Implementasi Grad-CAM (Gradient-weighted Class Activation Mapping)
    dari scratch menggunakan tf.GradientTape.

    Referensi: Selvaraju et al. (2017)
    Target layer: Conv_1 (output 7x7x1280, layer terakhir MobileNetV2)
    Dipilih berdasarkan komparasi empiris 4 kandidat layer.
    """

    def __init__(self, model, base_model, target_layer_name="Conv_1"):
        self.model      = model
        self.base_model = base_model

        # Sub-model dari dalam base_model:
        # input base_model → [Conv_1 output, output base_model]
        target_layer = base_model.get_layer(target_layer_name)

        self.grad_model_base = tf.keras.Model(
            inputs  = base_model.inputs,
            outputs = [target_layer.output, base_model.output]
        )

        # Custom head layers (setelah base_model)
        # Layer 0 = input, Layer 1 = base_model, Layer 2+ = head
        self.head_layers = model.layers[2:]

    def compute_heatmap(self, image_array, class_idx=None):
        """
        Hitung heatmap Grad-CAM untuk gambar input.

        Args:
            image_array : np.array shape (1, 224, 224, 3), sudah dinormalisasi
            class_idx   : index kelas target (None = ambil prediksi tertinggi)

        Returns:
            heatmap     : np.array shape (7, 7), nilai [0, 1]
            predictions : np.array shape (2,), probabilitas per kelas
        """
        with tf.GradientTape() as tape:
            # Forward pass melalui base_model
            conv_outputs, base_out = self.grad_model_base(
                image_array, training=False
            )
            tape.watch(conv_outputs)

            # Forward pass melalui head layers
            x = base_out
            for layer in self.head_layers:
                x = layer(x, training=False)
            predictions = x

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            class_score = predictions[:, class_idx]

        # Hitung gradien skor kelas terhadap feature map
        grads        = tape.gradient(class_score, conv_outputs)

        # Global Average Pooling gradien → bobot kepentingan per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted sum feature map
        heatmap = tf.reduce_mean(
            conv_outputs[0] * pooled_grads, axis=-1
        )

        # ReLU + normalisasi → [0, 1]
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy(), predictions.numpy()[0]

    def overlay_heatmap(self, image_uint8, heatmap, alpha=0.4):
        """
        Resize heatmap dan overlay ke gambar original.

        Args:
            image_uint8 : np.array shape (224, 224, 3), nilai [0, 255]
            heatmap     : np.array shape (7, 7), nilai [0, 1]
            alpha       : transparansi heatmap (default 0.4)

        Returns:
            superimposed    : np.array gambar original + heatmap
            heatmap_resized : np.array heatmap yang sudah di-resize ke 224x224
        """
        heatmap_resized = cv2.resize(
            heatmap,
            (image_uint8.shape[1], image_uint8.shape[0])
        )
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(
            heatmap_colored, cv2.COLOR_BGR2RGB
        )
        superimposed = cv2.addWeighted(
            image_uint8,     1 - alpha,
            heatmap_colored, alpha,
            0
        )
        return superimposed, heatmap_resized

    def preprocess_image(self, image_input):
        """
        Preprocessing gambar untuk inferensi.
        Menerima path file (str) atau PIL.Image atau np.array.

        Returns:
            img_input   : np.array shape (1, 224, 224, 3), dinormalisasi
            img_display : np.array shape (224, 224, 3), nilai [0, 255]
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.fromarray(image_input).convert("RGB")

        img         = img.resize(IMG_SIZE)
        img_display = np.array(img)

        img_norm  = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_display.astype("float32")
        )
        img_input = np.expand_dims(img_norm, axis=0)

        return img_input, img_display

    def explain(self, image_input, class_idx=None,
                true_label=None, show=False):
        """
        Pipeline lengkap: preprocessing → Grad-CAM → overlay.

        Args:
            image_input : path file (str), PIL.Image, atau np.array
            class_idx   : index kelas target (None = auto dari prediksi)
            true_label  : label asli untuk evaluasi (str atau int, opsional)
            show        : tampilkan visualisasi matplotlib (default False)

        Returns:
            dict berisi semua hasil analisis
        """
        img_input, img_display = self.preprocess_image(image_input)

        heatmap, predictions = self.compute_heatmap(img_input, class_idx)

        superimposed, heatmap_resized = self.overlay_heatmap(
            img_display, heatmap
        )

        pred_idx   = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx])

        # Evaluasi benar/salah jika true_label disediakan
        if true_label is not None:
            if isinstance(true_label, str):
                true_idx = CLASS_NAMES.index(true_label)
            else:
                true_idx = int(true_label)
            is_correct = (pred_idx == true_idx)
        else:
            is_correct = None

        # Visualisasi matplotlib — hanya untuk notebook/debug
        if show:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_display)
            axes[0].set_title(
                f"Gambar Original\n"
                f"Label Asli: {true_label.upper() if true_label else 'N/A'}",
                fontweight="bold"
            )
            axes[0].axis("off")

            im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title(
                "Grad-CAM Heatmap\nMerah=Penting | Biru=Tidak Penting",
                fontweight="bold"
            )
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, label="Tingkat Aktivasi")

            axes[2].imshow(superimposed)
            axes[2].set_title(
                f"Overlay Grad-CAM\n"
                f"Prediksi: {pred_class.upper()} ({confidence:.1%})",
                fontweight="bold",
                color="green" if is_correct else
                    "red"   if is_correct is False else "black"
            )
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()

        return {
            "predicted_class"   : pred_class,
            "confidence"        : confidence,
            "is_correct"        : is_correct,
            "all_probabilities" : {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            },
            "heatmap"           : heatmap,
            "heatmap_resized"   : heatmap_resized,
            "superimposed"      : superimposed,
            "image_display"     : img_display,
            "img_input"         : img_input,
        }


# ============================================================
# CLASS 2: GuidedGradCAM
# Sumber: Cell G-7 REVISED + Cell G-8 (versi fix)
# CATATAN PENTING: Implementasi ini menggunakan
# Input Gradient × Grad-CAM sebagai approximation,
# BUKAN Guided Backpropagation murni.
# Alasan: MobileNetV2 menggunakan Depthwise Separable
# Convolution yang tidak kompatibel dengan guided backprop.
# Hasil: saliency map yang dihasilkan tidak bermakna
# secara klinis (terdokumentasi sebagai keterbatasan).
# TIDAK ditampilkan di antarmuka Streamlit.
# ============================================================

class GuidedGradCAM:
    """
    Guided Grad-CAM — approximation berbasis Input Gradient × Grad-CAM.

    PERINGATAN: Tidak efektif pada MobileNetV2 karena Depthwise
    Separable Convolution. Dipertahankan untuk dokumentasi
    dan analisis di Bab 4 skripsi.
    """

    def __init__(self, model, base_model, target_layer_name="Conv_1"):
        self.model       = model
        self.base_model  = base_model
        self.head_layers = model.layers[2:]

        target_layer = base_model.get_layer(target_layer_name)
        self.grad_model_base = tf.keras.Model(
            inputs  = base_model.inputs,
            outputs = [target_layer.output, base_model.output]
        )

    def compute_gradcam(self, image_array, class_idx=None):
        """Hitung Grad-CAM biasa (sama dengan class GradCAM)."""
        with tf.GradientTape() as tape:
            conv_outputs, base_out = self.grad_model_base(
                image_array, training=False
            )
            tape.watch(conv_outputs)
            x = base_out
            for layer in self.head_layers:
                x = layer(x, training=False)
            predictions = x

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        grads        = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(
            conv_outputs[0] * pooled_grads, axis=-1
        )
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy(), predictions.numpy()[0], int(class_idx)

    def compute_saliency(self, image_array, class_idx=None):
        """
        Hitung Input Gradient × Input Image sebagai saliency map.
        Ini bukan Guided Backpropagation murni.
        """
        img_tensor = tf.cast(image_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            x = img_tensor
            for layer in self.model.layers[1:]:
                x = layer(x, training=False)
            predictions = x

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        grads    = tape.gradient(class_score, img_tensor)[0].numpy()
        saliency = np.abs(grads * image_array[0])
        saliency = np.max(saliency, axis=-1)

        saliency -= saliency.min()
        if saliency.max() > 0:
            saliency /= saliency.max()

        return saliency

    def explain_guided(self, image_input, true_label=None, show=False):
        """
        Pipeline Guided Grad-CAM.

        Returns dict berisi heatmap, saliency, guided_gradcam, overlay.
        """
        # Preprocess
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.fromarray(image_input).convert("RGB")

        img         = img.resize(IMG_SIZE)
        img_display = np.array(img)
        img_norm    = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_display.astype("float32")
        )
        img_input = np.expand_dims(img_norm, axis=0)

        # Grad-CAM
        heatmap, predictions, class_idx = self.compute_gradcam(img_input)

        # Saliency
        saliency = self.compute_saliency(img_input, class_idx)

        # Resize heatmap → 224×224
        heatmap_224 = cv2.resize(heatmap, (224, 224))

        # Guided Grad-CAM = heatmap × saliency
        guided_gc = heatmap_224 * saliency
        guided_gc -= guided_gc.min()
        if guided_gc.max() > 0:
            guided_gc /= guided_gc.max()

        # Contrast stretching (Cell G-8 fix)
        threshold = guided_gc.max() * 0.1
        guided_gc[guided_gc < threshold] = 0
        if guided_gc.max() > 0:
            guided_gc = (guided_gc - guided_gc.min()) / \
                        (guided_gc.max() - guided_gc.min() + 1e-8)
        guided_gc = np.power(guided_gc, 0.4)

        # Info prediksi
        pred_idx   = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx])

        if true_label is not None:
            true_idx = CLASS_NAMES.index(true_label) \
                if isinstance(true_label, str) else int(true_label)
            is_correct = (pred_idx == true_idx)
        else:
            is_correct = None

        # Overlay Grad-CAM
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_224), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay_gc = cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)

        # Overlay Guided Grad-CAM
        guided_colored = cv2.applyColorMap(
            np.uint8(255 * guided_gc), cv2.COLORMAP_JET
        )
        guided_colored = cv2.cvtColor(guided_colored, cv2.COLOR_BGR2RGB)
        overlay_guided = cv2.addWeighted(img_display, 0.45, guided_colored, 0.55, 0)

        return {
            "predicted_class" : pred_class,
            "confidence"      : confidence,
            "is_correct"      : is_correct,
            "all_probabilities": {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            },
            "heatmap"         : heatmap_224,
            "saliency"        : saliency,
            "guided_gradcam"  : guided_gc,
            "overlay_gradcam" : overlay_gc,
            "overlay_guided"  : overlay_guided,
            "image_display"   : img_display,
        }