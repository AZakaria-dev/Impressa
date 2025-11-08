Image to Lithophane STL

A clean Flask web app that:
1. Lets Users to upload an image and lets Users adjust brightness/contrast/gamma/sharpen/denoise
2. Gives a preview of a grayscale version adjusted through either default or user specified values for brightness/contrast/gamma/sharpen/denoise
3. Optionally asks **Gemini** for smart enhancement suggestions (JSON-only). User must provide an API key if you want this. (Somtimes doesn't work, will try to fix later)
4. Converts the adjusted image to a heightmap with the rule "darker is higher" or "lighter is higher" chosen by the User
5. Applies optional Gaussian smoothing to the heightmap.
6. Generates a closed STL (top surface + base + sidewalls) ready to be 3d pritned with Users target width, depth, and base thickness.
7. Lets you download the STL.

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
- Use White fillament when printing.
- All example images are printed in Elgoo white PLA on Bambu Labs X1 Carbon and P1S

## Output

STLs are written into the exports/ folder and also provided as a download link in the UI.
