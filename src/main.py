import numpy as np
from PIL import Image
from fastapi import FastAPI
import os
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    image = get_image()
    books = identify_books(image)
    recommendations = generate_recommendations(books)
    return {"recommended books":recommendations}

def get_image():
    input_image_path = os.path.join('test-data','book_shelf.jpg')
    image_object = Image.open(input_image_path)
    image_np = np.array(image_object) #RGB channel values per pixel
    return None

def identify_books(image):
    books = None
    return books

def generate_recommendations(books):
    recommendations = {}
    return recommendations



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)