from google import genai

client = genai.Client(api_key="AIzaSyASc7Jc65qSlhbCjopxERrFdn_QaGATdu4")
from google.genai import types

with open('4.png', 'rb') as f:
      image_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      'Identify all the visible text in the given image. And just return the text nothing else'
    ]
  )

print(response.text)