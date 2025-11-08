import os
import io
import base64
import traceback
from datetime import datetime
from typing import Tuple, Dict, Any
import google.generativeai as genai
print("google-generativeai version:", genai.__version__)
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename


# ---------------- Gemini integration (Still in Development) ----------------
def try_gemini_suggestions(gray_img: Image.Image, api_key: str) -> Dict[str, Any]:
    """
    Ask Gemini to analyze the grayscale image and propose advanced enhancement parameters
    that emphasize major details, boost contrast, and optimize local structure for heightmaps.
    """
    defaults = {"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "sharpen": 0.0, "denoise": 0.0}
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
            )

        # Downscale to make inference lighter
        preview = gray_img.copy()
        preview.thumbnail((256, 256))
        buf = io.BytesIO()
        preview.save(buf, format="PNG")

        # 🧠 Improved prompt for advanced contrast and detail mapping
        prompt = """
Analyze this grayscale image to prepare it for 3D heightmap generation.
Your goal: make major structures stand out, enhance contrast for depth clarity,
and emphasize important shapes while smoothing noise and minor gradients.
You may increase contrast strongly if it helps focus and definition.
Output ONLY a JSON object in this format:
{
 "brightness": float,  // 0.6–1.6
 "contrast": float,    // 0.8–2.5 (you may go higher than 1.8)
 "gamma": float,       // 0.5–1.6
 "sharpen": float,     // 0.0–1.0
 "denoise": float      // 0.0–1.5
}
Pick values that bring out edges and meaningful regions clearly for 3D printing.
Do not include any other text.
"""

        content = [
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}}
            ]}
        ]
        resp = model.generate_content(content)

        import json
        cfg = json.loads(resp.text.strip())

        # Apply Gemini’s parameters, with stronger bias toward contrast/detail
        defaults["brightness"] = float(cfg.get("brightness", defaults["brightness"]))
        defaults["contrast"]   = min(float(cfg.get("contrast", defaults["contrast"])) * 1.2, 2.5)
        defaults["gamma"]      = float(cfg.get("gamma", defaults["gamma"]))
        defaults["sharpen"]    = min(float(cfg.get("sharpen", defaults["sharpen"])) + 0.3, 1.0)
        defaults["denoise"]    = max(float(cfg.get("denoise", defaults["denoise"])) - 0.1, 0.0)

        # Post-process: add localized focus contrast
        # (this runs later automatically via apply_adjustments)
    except Exception:
        traceback.print_exc()
    return defaults

# ---------------- Image ops ----------------
def apply_adjustments(gray: Image.Image, brightness: float, contrast: float, gamma: float,
                      sharpen: float, denoise: float) -> Image.Image:
    if brightness != 1.0:
        gray = ImageEnhance.Brightness(gray).enhance(brightness)
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    if abs(gamma - 1.0) > 1e-3:
        arr = np.asarray(gray).astype(np.float32) / 255.0
        arr = np.clip(arr, 0, 1) ** gamma
        gray = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    if denoise > 0.0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=min(denoise * 1.5, 8.0)))
    if sharpen > 0.0:
        gray = gray.filter(ImageFilter.UnsharpMask(
            radius=min(1.5 + 1.5 * sharpen, 5.0),
            percent=int(150 + 100 * sharpen),
            threshold=3
        ))
    return gray


def gaussian_blur_heightmap(heightmap: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return heightmap
    radius = int(max(1, min(int(sigma * 3), 30)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=heightmap)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=tmp)


def write_ascii_stl(path: str, vertices: np.ndarray, faces: np.ndarray):
    def tri_normal(a, b, c):
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        return n / norm if norm != 0 else np.array([0.0, 0.0, 0.0])
    with open(path, "w") as f:
        f.write("solid heightmap\n")
        for (ia, ib, ic) in faces:
            a, b, c = vertices[ia], vertices[ib], vertices[ic]
            n = tri_normal(a, b, c)
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in (a, b, c):
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid heightmap\n")


def heightmap_to_mesh(z: np.ndarray, width_mm: float, depth_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed, printable solid: top + bottom + all four side walls."""
    H, W = z.shape
    xs = np.linspace(0, width_mm, W, dtype=np.float32)
    ys = np.linspace(0, depth_mm, H, dtype=np.float32)
    Y_corrected = np.flipud(ys)  # fix slicer orientation
    X, Y = np.meshgrid(xs, Y_corrected)

    top_vertices = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    bottom_vertices = np.stack([X.ravel(), Y.ravel(), np.zeros_like(z).ravel()], axis=1)
    vertices = np.vstack([top_vertices, bottom_vertices])

    def idx_top(y, x): return y * W + x
    base_offset = H * W
    def idx_bot(y, x): return base_offset + y * W + x

    faces = []
    # Top + bottom
    for y in range(H - 1):
        for x in range(W - 1):
            v00 = idx_top(y, x); v10 = idx_top(y + 1, x)
            v01 = idx_top(y, x + 1); v11 = idx_top(y + 1, x + 1)
            faces.append([v00, v10, v01]); faces.append([v01, v10, v11])

            v00b = idx_bot(y, x); v10b = idx_bot(y + 1, x)
            v01b = idx_bot(y, x + 1); v11b = idx_bot(y + 1, x + 1)
            faces.append([v01b, v10b, v00b]); faces.append([v11b, v10b, v01b])

    # Side walls (perimeter)
    def wall(a1, a2, b1, b2):
        # triangles oriented outward
        faces.append([a1, a2, b1])
        faces.append([b1, a2, b2])

    # front (y=0)
    for x in range(W - 1):
        wall(idx_top(0, x), idx_top(0, x + 1), idx_bot(0, x), idx_bot(0, x + 1))
    # back (y=H-1)
    for x in range(W - 1):
        wall(idx_top(H - 1, x + 1), idx_top(H - 1, x), idx_bot(H - 1, x + 1), idx_bot(H - 1, x))
    # left (x=0)
    for y in range(H - 1):
        wall(idx_top(y + 1, 0), idx_top(y, 0), idx_bot(y + 1, 0), idx_bot(y, 0))
    # right (x=W-1)
    for y in range(H - 1):
        wall(idx_top(y, W - 1), idx_top(y + 1, W - 1), idx_bot(y, W - 1), idx_bot(y + 1, W - 1))

    return vertices.astype(np.float32), np.array(faces, dtype=np.int32)


def to_heightmap(gray: Image.Image, darker_is_higher: bool, height_mm: float, base_mm: float,
                 max_dim: int = 350) -> np.ndarray:
    """Uniform downscale to max_dim; preserve aspect; add 1px border earlier in pipeline."""
    w, h = gray.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(gray).astype(np.float32) / 255.0
    norm = 1.0 - arr if darker_is_higher else arr
    return base_mm + norm * height_mm


# ---------------- Flask ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "devkey")

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EDITED_DIR = os.path.join(BASE_DIR, "edited")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
for d in [UPLOAD_DIR, EDITED_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    try:
        if "image" not in request.files:
            flash("Please choose an image.", "error");  return redirect(url_for("index"))

        file = request.files["image"]
        if not file or not file.filename:
            flash("Please choose an image.", "error");  return redirect(url_for("index"))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = secure_filename(file.filename)
        safe_name = f"{ts}_{base}"
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        file.save(upload_path)

        params = {k: request.form.get(k) for k in request.form.keys()}
        gemini_key = request.form.get("gemini_key") or os.environ.get("GOOGLE_API_KEY")

        # open + border
        img = Image.open(upload_path).convert("RGB")
        img = ImageOps.expand(img, border=1, fill="black")
        gray = ImageOps.grayscale(img)

        # defaults: scale to ~100mm bounding box, keep aspect
        px_w, px_h = img.size
        max_size_mm = 100.0
        scale = max_size_mm / max(px_w, px_h)
        params["default_width_mm"] = round(px_w * scale, 2)
        params["default_depth_mm"] = round(px_h * scale, 2)

        # manual or Gemini adjustments
        brightness = float(params.get("brightness", 1.0))
        contrast   = float(params.get("contrast", 1.0))
        gamma      = float(params.get("gamma", 1.0))
        sharpen    = float(params.get("sharpen", 0.0))
        denoise    = float(params.get("denoise", 0.0))

        if gemini_key and params.get("use_gemini") == "on":
            sugg = try_gemini_suggestions(gray, gemini_key)
            brightness, contrast, gamma, sharpen, denoise = (
                sugg["brightness"], sugg["contrast"], sugg["gamma"], sugg["sharpen"], sugg["denoise"]
            )

        gray = apply_adjustments(gray, brightness, contrast, gamma, sharpen, denoise)

        # gentle global tune to match your "manual" look
        gray = gray.filter(ImageFilter.GaussianBlur(radius=0.6))
        arr = np.asarray(gray).astype(np.float32) / 255.0
        arr = np.power(arr, 0.9)
        gray = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        gray = ImageOps.autocontrast(gray, cutoff=2)

        # save edited image and pass the reference forward
        edited_name = f"edited_{safe_name}.png"
        edited_path = os.path.join(EDITED_DIR, edited_name)
        gray.save(edited_path)
        params["edited_file"] = edited_name  # <-- MUST be present

        # preview b64
        buf = io.BytesIO();  gray.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return render_template("preview.html", preview_b64=b64, params=params)
    except Exception as e:
        traceback.print_exc()
        flash(f"Preview error: {e}", "error")
        return redirect(url_for("index"))


@app.route("/generate", methods=["POST"])
def generate():
    try:
        edited_name = request.form.get("edited_file", "").strip()
        if not edited_name:
            flash("Internal error: missing edited image reference. Please re-upload.", "error")
            return redirect(url_for("index"))

        file_path = os.path.join(EDITED_DIR, secure_filename(edited_name))
        if not os.path.exists(file_path):
            flash("Edited image not found. Please re-upload.", "error")
            return redirect(url_for("index"))

        params = {k: request.form.get(k) for k in request.form.keys()}
        gray = ImageOps.grayscale(Image.open(file_path))

        # dims (height fixed default 2.5; user can change)
        height_mm = float(params.get("height_mm", 2.5))
        base_mm   = float(params.get("base_mm", 0.4))
        width_mm  = float(params.get("width_mm")  or params.get("default_width_mm")  or 100.0)
        depth_mm  = float(params.get("depth_mm")  or params.get("default_depth_mm")  or 100.0)
        darker_is_higher = (params.get("dark_higher", "dark") == "dark")
        smooth_sigma = float(params.get("smooth", 0.0))
        max_grid = int(params.get("max_grid", 300))

        z = to_heightmap(gray, darker_is_higher, height_mm, base_mm, max_dim=max_grid)
        if smooth_sigma > 0:
            z = gaussian_blur_heightmap(z, smooth_sigma)

        V, F = heightmap_to_mesh(z, width_mm, depth_mm)

        stl_name = f"heightmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
        stl_path = os.path.join(EXPORT_DIR, stl_name)
        write_ascii_stl(stl_path, V, F)

        return render_template("success.html", stl_name=stl_name)
    except Exception as e:
        traceback.print_exc()
        flash(f"Generation error: {e}", "error")
        return redirect(url_for("index"))


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # You can change port via PORT env var if desired
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=True)
