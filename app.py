# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from caries_model import detect_caries_and_teeth
# import os
#
# app = FastAPI()
#
# # Add CORS middleware to allow requests from Angular frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:4200"],  # Your Angular app's URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Root endpoint for testing
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Caries Detection API! Use POST /detect_caries to upload an image."}
#
# # Caries detection endpoint (handles both /detect_caries and /detect_caries/)
# @app.post("/detect_caries")
# @app.post("/detect_caries/")
# async def detect_caries(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded file temporarily
#         file_path = f"temp_{file.filename}"
#         with open(file_path, "wb") as buffer:
#             buffer.write(await file.read())
#
#         # Run the caries detection using the models
#         result = detect_caries_and_teeth(
#             image_input=file_path,
#             yolo_model_path='C:/Users/marit/Desktop/ANUL4/LICENTA/Proiect/yolo_runs/train5/weights/best.pt',
#             caries_model_path='C:/Users/marit/Desktop/ANUL4/LICENTA/Proiect/models/best_model4.pth',
#             threshold=0.01
#         )
#
#         # Clean up the temporary file
#         os.remove(file_path)
#
#         # Return the detection results
#         return {
#             "output_image": result["output_image"],
#             "caries_teeth": result["caries_teeth"],
#             "caries_area_ratio": result["caries_area_ratio"]
#         }
#     except Exception as e:
#         # Handle errors (e.g., file not found, model loading issues)
#         return {"error": str(e)}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# /home/annamarita/mysite/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from caries_model import detect_caries_and_teeth
import os

app = FastAPI()

# Add CORS middleware to allow requests from Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://annamarita.pythonanywhere.com"],  # Update for hosted URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Caries Detection API! Use POST /detect_caries to upload an image."}

# Caries detection endpoint (handles both /detect_caries and /detect_caries/)
@app.post("/detect_caries")
@app.post("/detect_caries/")
async def detect_caries(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Run the caries detection using the models
        result = detect_caries_and_teeth(
            image_input=file_path,
            yolo_model_path='/app/models/best.pt',
            caries_model_path='/app/models/best_model4.pth',
            threshold=0.01
        )

        # Clean up the temporary file
        os.remove(file_path)

        # Return the detection results
        return {
            "output_image": result["output_image"],
            "caries_teeth": result["caries_teeth"],
            "caries_area_ratio": result["caries_area_ratio"]
        }
    except Exception as e:
        # Handle errors (e.g., file not found, model loading issues)
        return {"error": str(e)}