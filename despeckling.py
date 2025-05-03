from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import base64
from methods import makedirs, extract_bands, extract_patches, despeckle_image, reconstruct_image, recombine_bands, load_model

upload_folder = "upload"
input_band_folder = "input_band"
input_patch_folder = "patch"
despeckled_patch_folder = "despeckled_patch"
despeckled_band_folder = "despeckled_band"
result_folder = "result"

dir_list = [upload_folder, input_band_folder, input_patch_folder, despeckled_patch_folder, despeckled_band_folder, result_folder]


model = load_model()
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def make_pred():
    input_json = request.get_json()
    input_imageb64 = input_json["image"]
    height = input_json["height"]
    width = input_json["width"]
    band = input_json["band"]
    #type = input_json["type"]
    name = "input"

    # Decode base64 and interpret as uint16
    input_image_bytes = base64.b64decode(input_imageb64)
    input_image = np.frombuffer(input_image_bytes, dtype= np.uint16).reshape((band, height, width))
    type = input_image.dtype


    makedirs(dir_list)
    extract_bands(input_image, input_band_folder, name)
    
    extract_patches(input_band_folder, 2745, save_directory  = input_patch_folder)

    despeckle_image(model, input_patch_folder, despeckled_patch_folder)

    reconstruct_image(despeckled_patch_folder, despeckled_band_folder)

    output_image = recombine_bands(despeckled_band_folder, type)


    output_image_bytes = np.array(output_image).astype(np.uint16).tobytes()
    output_imageb64 = base64.b64encode(output_image_bytes).decode("utf-8")

    return jsonify({"image": output_imageb64})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)