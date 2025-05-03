import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import time

# Simulate an image
with rasterio.open("input.tif") as src:
    input_image = src.read()

#input_image = input_image[:, :64, :64]
image = input_image.tobytes()
imageb64 = base64.b64encode(image).decode("utf-8")


params = {
    "image": imageb64,
    "height": int(input_image.shape[1]),
    "width": int(input_image.shape[2]),
    "band": int(input_image.shape[0]),
    #"type": input_image.dtype,

}
print(params['height'], params['width'])
url = "http://127.0.0.1:5000/predict"

# POST request
pred = requests.post(url, json=params)

# ✅ Properly get image back from JSON response
imageb64 = pred.json()["image"]

# Decode base64 string to bytes
image = base64.b64decode(imageb64)

# Convert to NumPy array
image = np.frombuffer(image, np.uint16)
image = image.reshape((input_image.shape))  # Assuming you want it back in original shape

print("response received")
plt.figure(figsize=(10, 20))
plt.subplot(1, 2, 1)
plt.imshow(input_image[0])#, cmap = "gray")
plt.title("Input Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image[0])#, cmap="gray")
plt.title("Despeckled Image")
plt.axis("off")
plt.show()
