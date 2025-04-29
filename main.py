import io
import os
from pathlib import Path
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import subprocess

os.environ["RECOGNITION_BATCH_SIZE"] = "32"
os.environ["DETECTOR_BATCH_SIZE"] = "4"

from extractor import LabReportExtractor

class LabTestData(BaseModel):
    test_name: str
    test_value: Optional[str] = None
    bio_reference_range: Optional[str] = None
    test_unit: Optional[str] = None
    lab_test_out_of_range: Optional[bool] = None

class LabReportResponse(BaseModel):
    is_success: bool
    data: Optional[List[LabTestData]] = None
    error: Optional[str] = None

app = FastAPI(
    title="Lab Report Parser API",
    description="Extracts lab test data from lab report images using Surya OCR",
    version="1.0.0"
)

extractor = LabReportExtractor()

detection_predictor = None
recognition_predictor = None

@app.on_event("startup")
async def load_models():
    global detection_predictor, recognition_predictor
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor()

@app.post("/get-lab-tests", response_model=LabReportResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    global detection_predictor, recognition_predictor, extractor
    temp_file_path = None
    output_dir = None
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or '.tmp') as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    ocr_lines_to_process = None
    output_dir = tempfile.mkdtemp()
    result_path = os.path.join(output_dir, "results.json")
    subprocess.run([
        "surya_ocr", temp_file_path, "--langs", "en", "--output_dir", output_dir
    ], capture_output=True, text=True, check=True, timeout=60)
    if os.path.exists(result_path):
        with open(result_path, 'r') as f: ocr_data = json.load(f)
        base_name = Path(temp_file_path).stem
        if base_name in ocr_data and ocr_data[base_name]:
            page_result = ocr_data[base_name][0]
            if 'text_lines' in page_result:
                ocr_lines_to_process = page_result['text_lines']
    if ocr_lines_to_process is None:
        image = Image.open(temp_file_path).convert("RGB")
        detection_results = detection_predictor([image])
        if detection_results and detection_results[0].bboxes:
            recognition_results = recognition_predictor(
                [image], [["en"]], detection_predictor
            )
            if recognition_results and recognition_results[0].text_lines:
                ocr_lines_to_process = [
                    {"text": line.text, "bbox": line.bbox, "confidence": line.confidence}
                    for line in recognition_results[0].text_lines
                ]
    lab_tests = extractor.process_ocr_results(ocr_lines_to_process)
    validated_tests = [LabTestData(**test) for test in lab_tests]
    os.unlink(temp_file_path)
    import shutil
    shutil.rmtree(output_dir)
    return LabReportResponse(is_success=True, data=validated_tests)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)