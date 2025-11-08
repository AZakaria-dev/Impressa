Image to Lithophane STL

A clean Flask web app that:
1. Lets you upload an image and preview a grayscale version with adjustable brightness/contrast/gamma/sharpen/denoise.
2. Optionally asks Gemini for smart enhancement suggestions (JSON-only). User must provide an API key if you want this. (Somtimes doesn't work, will try to fix later)
3. Converts the adjusted image to a heightmap with the rule "darker is higher" or "lighter is higher"
4. Applies optional Gaussian smoothing to the heightmap.
5. Generates a closed STL (top surface + base + sidewalls) with Users target width, depth, and base thickness.
6. Lets you download the STL.

No virtual environment required (though recommended). The STL writer is pure Python (ASCII STL).

## Install

```bash
# (optional) python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export FLASK_SECRET_KEY=dev
python app.py
```

Then open http://localhost:5001

## Usage Tips

- **Max grid**: The app downsamples large images to keep mesh sizes reasonable. Increase if you want a denser mesh.
- **Height rule**: Choose whether darker or lighter regions should be higher in the relief.
- **Smoothing**: Gaussian σ in pixels after downsampling. Start with 0.6–1.2 for photos.
- **Dimensions**: Width/Depth set XY real-world size in millimeters. Base sets the plate thickness.
- **Gemini**: If enabled and the key is present, the app uploads a small preview to Gemini and applies the returned adjustments.

## Output

STLs are written into the exports/ folder and also provided as a download link in the UI.
