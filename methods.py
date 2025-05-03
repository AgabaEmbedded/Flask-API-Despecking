import os, numpy as np
from PIL import Image
import time
import shutil
import tensorflow as tf
import requests

def load_model():

    url = "https://github.com/AgabaEmbedded/Large_files/releases/download/v1.0/Large.model.keras"
    response = requests.get(url)

    with open("model.keras", "wb") as f:
        f.write(response.content)
    model = tf.keras.models.load_model("model.keras")
    return model

def makedirs(dir_list):
    
    for dir in dir_list:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

#function to extract image bands
def extract_bands(image, output_folder, image_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for band in range(0, len(image)):
        band_data = image[band]

        if band_data.dtype == np.uint8:
            band_data = band_data.astype(np.uint16)
            #band_data = ((band_data-band_data.min())/(band_data.max()-band_data.min())*65535).astype(np.uint16)
            #band_data = (band_data/255*65535).astype(np.uint16)
        else:
            band_data = band_data.astype(np.uint16)
            #band_data = ((band_data-band_data.min())/(band_data.max()-band_data.min())*65535).astype(np.uint16)

        output_filename = f"{image_name}_{band}.png"
        output_path = os.path.join(output_folder, output_filename)

        Image.fromarray(band_data).save(output_path)
        print(f"Saved: {output_filename}")


def extract_patches(image_folder, patch_size, save_directory, stride = 4):
  for image_dir in sorted(os.listdir(image_folder)):

    image_path = os.path.join(image_folder, image_dir)
    image_name = image_dir.replace(".png", "")
    image = Image.open(image_path)
    image = np.array(image)
    h, w = image.shape




    for i in range(0, h, (patch_size-stride)):
      for j in range(0, w, patch_size-stride):
        end_i = min(i+patch_size, h)
        end_j = min(j+patch_size, w)
        patch = image[i:end_i, j:end_j]

        patch = Image.fromarray(patch.astype(np.uint16))
        save_path = os.path.join(save_directory, f"{image_name}_{h}_{w}_{i}_{j}.png")
        patch.save(save_path)

    print(f"done patching image {image_name}")




# Function to tile image into overlapping patches
def despeckle_image(model, input_image_folder, output_image_folder, patch_size=64, edge_crop=4):
    start_time = time.time()
    stride = patch_size - 2 * edge_crop  # Ensure overlap
    count = 1

    for image_path in sorted(os.listdir(input_image_folder)):
        
        print(f"patch {count}/{len(os.listdir(input_image_folder))}")

        image_name = image_path.split(".")[0]
        
        image_path = os.path.join(input_image_folder, image_path)

        image = Image.open(image_path)
        image = np.array(image).astype(np.float32) / 65535.0
        h, w = image.shape
        patches, positions = [], []

        # Extract overlapping patches
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                patch = image[i:i+patch_size, j:j+patch_size]

                pad_h = max(0, patch_size - patch.shape[0])
                pad_w = max(0, patch_size - patch.shape[1])

                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')

                patches.append(patch)
                positions.append((i + edge_crop, j + edge_crop))  # Place cropped center patch

        patches = np.expand_dims(patches, axis=-1)  # Add channel dimension

        # Model prediction
        sr_patches = model.predict(patches, batch_size = 64)
        if isinstance(sr_patches, dict):
            sr_patches = sr_patches["output_0"]

        # Initialize stitched image and weight map
        reconstructed = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        valid_area = patch_size - 2 * edge_crop


        for sr_patch, (i, j) in zip(sr_patches, positions):
            valid_patch = sr_patch[edge_crop:-edge_crop, edge_crop:-edge_crop, 0]  # Expected size: 56x56

            # Calculate boundaries
            dst_patch_h = min(valid_patch.shape[0], h - i)
            dst_patch_w = min(valid_patch.shape[1], w - j)

            # Skip invalid patches
            if dst_patch_h <= 0 or dst_patch_w <= 0:
                continue

            valid_patch = valid_patch[:dst_patch_h, :dst_patch_w]

            reconstructed[i:i+dst_patch_h, j:j+dst_patch_w] += valid_patch
            weight_map[i:i+dst_patch_h, j:j+dst_patch_w] += 1



        # Normalize overlapping regions
        weight_map[weight_map == 0] = 1
        reconstructed = reconstructed / weight_map
        reconstructed = (reconstructed * 65535).clip(0, 65535).astype(np.uint16)

        output_path = os.path.join(output_image_folder, f"{image_name}_out.png")
        Image.fromarray(reconstructed).save(output_path)
        count += 1
    end_time = time.time()
    print(f"time: {end_time-start_time} seconds")



def reconstruct_image(patch_folder, output_path, stride = 4):
    previous_image_name = None
    reconstructed_image = None

    for patch_file in sorted(os.listdir(patch_folder)):
        if patch_file.endswith('.png'):
            # Extract parts from filename
            
            parts = patch_file.rsplit('_', 5)
            

            width = int(parts[-4])
            height = int(parts[-5])
            i = int(parts[-3])
            j = int(parts[-2])
            image_name = parts[0]

            # Save the previous image if we've moved on to a new one
            if previous_image_name is not None and image_name != previous_image_name:
                image_save_path = os.path.join(output_path, f"{previous_image_name}_out.png")
                Image.fromarray(reconstructed_image.astype(np.uint16)).save(image_save_path)
                print(f"Reconstructed image saved to {image_save_path}")
                reconstructed_image = None  # Reset for next image

            # Initialize new image array if needed
            if reconstructed_image is None:
                reconstructed_image = np.zeros((height, width), dtype=np.float32)

            # Load patch and place it
            patch_path = os.path.join(patch_folder, patch_file)
            patch = np.array(Image.open(patch_path))
            patch_height, patch_width = patch.shape

            #print(f"i: {i}, j: {j}, i+patch_heigtht: {i+patch_height}, j+patch_width: {j+patch_width}")
            reconstructed_image[(i+stride):i+patch_height, (j+stride):j+patch_width] = patch[stride:, stride:]
            previous_image_name = image_name

    # Save the last reconstructed image
    if reconstructed_image is not None:
        image_save_path = os.path.join(output_path, f"{previous_image_name}_out.png")
        Image.fromarray(reconstructed_image.astype(np.uint16)).save(image_save_path)
        print(f"Reconstructed image saved to {image_save_path}")


def recombine_bands(super_res_folder, type):
    #if not os.path.exists(final_output_folder):
     #   os.makedirs(final_output_folder)

    band_files = {}  # Dictionary to group bands by original filename

    for band_dir in os.listdir(super_res_folder):
        if band_dir.endswith("_out.png"):
            base_name = band_dir.rsplit('_', 2)[0]  # Extract original filename before suffixes
            band_num = band_dir.rsplit('_', 2)[1]

            if base_name not in band_files:
                band_files[base_name] = []

            band_files[base_name].append((band_num, os.path.join(super_res_folder, band_dir)))

    for base_name, band_list in band_files.items():
        band_list.sort()  # Ensure bands are in order
        band_arrays = []

        for _, band_path in band_list:
            band_data = np.array(Image.open(band_path))
            if type == "uint8":
                band_data = np.clip(band_data, 0, 255).astype(np.uint8)
                #band_data = (band_data/65535*255).astype(np.uint8)
                #band_data = ((band_data-band_data.min())/(band_data.max()-band_data.min())*255).astype(np.uint8)
            else:
                band_data = band_data.astype(np.uint16)
            band_arrays.append(band_data)

        #band_stack = np.stack(band_arrays, axis=0)  # Combine into a multi-band array
        #output_tiff = os.path.join(final_output_folder, f"{base_name}_out.tif")
        #with rasterio.open(output_tiff, "w", **metadata) as dst:
        #  for i, band in enumerate(band_arrays, start=1):
        #      dst.write(band, i)

        print(f"Saved recombined GeoTIFF: {base_name}")
        return band_arrays