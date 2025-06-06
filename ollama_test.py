import ollama
import base64
import os

image_path = "C:/Users/ILINT111/Desktop/test/rafter2.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' does not exist.")
else:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = ollama.chat(
            model="llava:7b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The image shows a rafter roof structure. The rafter is 10 meters long and inclined at a 60-degree angle. "
                        "There are multiple parallel wooden beams running across it. One of the wooden beams is a standard 2x10 beam "
                        "(5.08 cm wide and 25.4 cm tall), which you can use as a reference for scale. "
                        "Estimate the distance in centimeters between two adjacent wooden beams, based on this reference object. "
                        "The measurement does not need to be precise—just a reasonable estimate."
                        "the roof has a consistent pattern."
                    ),
                    "images": [encoded_image]
                }
            ]
        )

        print(response["message"]["content"])

    except Exception as e:
        print(f"Error during API call: {str(e)}")
