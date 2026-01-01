# SAR-Image-Despeckling-API

A lightweight **Flask-based REST API** for despeckling multi-band SAR (Synthetic Aperture Radar) images using a deep learning model. The service accepts a base64-encoded uint16 multi-band image, removes speckle noise band-by-band using overlapping patch processing, and returns the despeckled image in the same format.

Perfect for integration into remote sensing pipelines, web applications, or desktop tools that need on-demand SAR despeckling.

## Features

- Accepts **multi-band 16-bit (uint16) SAR images** via JSON POST request  
- Processes each band independently with a trained deep learning model  
- Uses **overlapping patch extraction + weighted reconstruction** to avoid border artifacts  
- Returns despeckled image as base64-encoded uint16 array (same shape and type as input)  
- Runs locally with Flask (easy to containerize)  
- Minimal dependencies  

## File Structure

```
.
├── README.md               # This file
├── despeckling.py          # Main Flask API server
├── methods.py              # Core processing functions (patching, despeckling, reconstruction)
├── model.keras             # Pre-trained Keras despeckling model (CNN/U-Net style)
├── requirements.txt        # Python dependencies
└── sendrequest.py          # Example client script to test the API
```

## Requirements

- Python 3.8+
- TensorFlow 2.x (with GPU support optional)
- A decent GPU is recommended for fast inference (CPU works but slower)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SAR-Image-Despeckling-API.git
   cd SAR-Image-Despeckling-API
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## requirements.txt

```txt
Flask>=2.0.0
tensorflow>=2.10.0
numpy
Pillow
```

> Note: TensorFlow version may vary depending on your system (CPU/GPU). Use `tensorflow` for CPU or `tensorflow-gpu` if available.

## Running the Server

```bash
python despeckling.py
```

The API will start on `http://0.0.0.0:5000`

## API Endpoint

### POST /predict

**Request Format (JSON):**
```json
{
  "image": "base64_encoded_string",
  "height": 1024,
  "width": 1024,
  "band": 4
}
```

- `image`: Base64-encoded raw bytes of the uint16 multi-band image (shape: band × height × width)
- `height`, `width`: Image dimensions
- `band`: Number of bands (channels)

**Response:**
```json
{
  "image": "base64_encoded_despeckled_image"
}
```

The returned image has the **same shape, dtype (uint16), and band order** as the input.

## Example Client Usage

Use the provided `sendrequest.py` to test:

```bash
python sendrequest.py path/to/your_sar_image.npy
```

The script loads a NumPy array (band × H × W, uint16), encodes it, sends to the API, and saves the result.

You can modify it easily for TIFF, ENVI, or other formats.

## How It Works (Processing Pipeline)

1. Decode base64 → reshape to (bands, height, width) uint16 NumPy array
2. Split into individual band images and save temporarily
3. Extract overlapping patches from each band
4. Run inference on patches using the loaded `model.keras`
5. Reconstruct each band using weighted averaging of overlapping regions
6. Recombine despeckled bands into original multi-band structure
7. Encode result back to base64 and return

Temporary folders (`upload/`, `input_band/`, etc.) are cleaned and recreated on each request.

## Model Details

- `model.keras`: A convolutional neural network trained specifically for SAR despeckling
- Input patch size: 64×64 (with edge cropping for seamless stitching)
- Overlap strategy ensures no visible seams in final output

## Performance Tips

- For large images (>5000×5000), consider batching requests or increasing patch stride
- GPU acceleration significantly speeds up inference
- The current patch size and overlap are optimized for quality vs speed

## Future Improvements (Contributions Welcome!)

- Support for direct TIFF/GeoTIFF input/output
- Batch processing endpoint
- Model quantization for faster CPU inference
- Async processing with queue
- Docker containerization
- OpenAPI/Swagger documentation

## License

MIT License – Feel free to use, modify, and distribute.

---

**Note**: This project is designed for research and operational use in SAR image processing. The included model is trained on synthetic and real SAR data for effective speckle reduction while preserving edges and textures.

Enjoy cleaner SAR imagery!
