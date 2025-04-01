from PIL import Image

# Open the image
image = Image.open("training_data/NG.jpg").convert("RGBA")  # Convert to RGBA mode

# Get pixel data
data = image.getdata()

# Define a threshold for "black-ish" colors
threshold = 50  # Any RGB values <= 50 are considered "black-ish"

# Process pixels
new_data = []
for item in data:
    r, g, b, a = item  # Extract RGBA channels
    # Check if the pixel is "black-ish"
    if r <= threshold and g <= threshold and b <= threshold:
        new_data.append((0, 0, 0, 0))  # Fully transparent
    else:
        new_data.append(item)

# Update image data
image.putdata(new_data)

# Save the result
image.save("output_image.png")
image.show()
