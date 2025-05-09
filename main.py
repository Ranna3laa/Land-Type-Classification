import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure necessary directories exist
os.makedirs("static/test", exist_ok=True)

# Load YOLO model
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is included in the project.")

model = YOLO(MODEL_PATH)
model.to("cpu")
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), request: Request = None):
    """Handles image upload, performs YOLO inference, and returns the result."""
    
    image_path = f"static/test/{file.filename}"
    with open(image_path, "wb") as image_file:
        image_bytes = await file.read()
        image_file.write(image_bytes)

    # Load image and convert to numpy
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Run YOLO model on the image
    results = model(img_np)
    output = results[0]

    # Draw bounding boxes if objects are detected
    draw = ImageDraw.Draw(img)  
    predicted_classes = []

    if not hasattr(output, "boxes") or len(output.boxes) == 0:
        return JSONResponse(content={
            "prediction": "No objects detected",
            "image_url": f"{request.base_url._url.rstrip('/')}/static/test/{file.filename}"
        })

    for box in output.boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]  
        predicted_classes.append(class_name)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{class_name}", fill="red")

    # Save processed image
    processed_filename = f"processed_{file.filename}"
    processed_image_path = f"static/test/{processed_filename}"
    img.save(processed_image_path)

    # Construct full URL for the processed image
    base_url = request.base_url._url.rstrip("/") if request else "http://127.0.0.1:8000"

    return JSONResponse(content={
        "prediction": predicted_classes,
        "image_url": f"{base_url}/static/test/{processed_filename}"
    })

# Run Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
