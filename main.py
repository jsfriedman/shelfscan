from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    image = get_image()
    books = identify_books(image)
    recommendations = generate_recommendations(books)
    return {"recommended books":recommendations}

def get_image():
    image = None
    return image

def identify_books(image):
    books = None
    return books

def generate_recommendations(books):
    recommendations = None
    return recommendations
