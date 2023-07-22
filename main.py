from fastapi import FastAPI

app = FastAPI()



@app.get("/")
async def root():
    image = None
    user = None
    book_spines = identify_book_spines(image)
    book_spines_text = determine_book_spines_text(book_spines)
    books = resolve_to_books(book_spines_text)
    books_to_recommend = recommend_books(books, user)
    return {"recommended books":books_to_recommend}


def identify_book_spines(image):
    book_spines = None
    return book_spines


def determine_book_spines_text(book_spines):
    book_spines_text = None
    return book_spines_text

def resolve_to_books(book_spines_text):
    books = None
    return books

def recommend_books(books, user):
    books_to_recommend = {}
    return books_to_recommend
