# Blurry Like Google Maps

Blur faces or license plates in a photo using a YOLO model and OpenCV.

## Motivation

I use Google Maps Street View and was curious about how they blur human faces, license plates, and other details. Just a quick playground 😁

## What it does

The script:

1. Downloads a pretrained face-detection model from Hugging Face.
2. Reads the input image.
3. Detects faces in the image.
4. Blurs each detected region.
5. Saves the anonymized image as `anonymized_traffic_cam.jpg`.

## Requirements

- Python 3.10+
- A virtual environment is recommended
- An input image named `6th-street-2500x750.jpg` in the project folder

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

After installing the dependencies, run:

```bash
python blur.py
```

The script will create `anonymized_traffic_cam.jpg` in the project directory.

## Changing the input image

If you want to use a different image, open `blur.py` and replace the hardcoded file name:

```python
results = model.predict("6th-street-2500x750.jpg", save=True)
img = cv2.imread("6th-street-2500x750.jpg")
```

Update both lines to point to your image.

## Notes

- The first run may take longer because the model is downloaded from Hugging Face.
- `opencv-python-headless` is used so the project can run without GUI dependencies.
- If you want the packages fully reproducible, the versions in `requirements.txt` are pinned.
