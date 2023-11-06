import os

import easyocr
import numpy as np
import pandas as pd
import rasterio.features
import uvicorn
from PIL import Image
from fastapi import FastAPI
from scipy.ndimage import rotate as scipy_rotate
from shapely import Polygon
from ultralytics import YOLO
from dotenv import load_dotenv
import requests

import time

load_dotenv()

tic = time.perf_counter()
app = FastAPI()
print('starting server')
model = YOLO(r"yolov8_n_24aug2023.pt")
print('yolo model loaded')
reader = easyocr.Reader(['en'])
print('ocr model loaded')
ngram_data_df = pd.read_csv('../3grams.csv')
print('ngram data loaded')

GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
toc = time.perf_counter()
print(f"server set-up time: {toc-tic:0.2f} seconds")


@app.get("/")
async def root():
    tic = time.perf_counter()
    input_image = Image.open(r"..\test-data\book_shelf3.jpg")

    results = model.predict(source=input_image, save=True, show_labels=False, show_conf=False, boxes=False)
    bookspine_masks = [_ for _ in results[0].masks.xy if len(_) > 0]  # filter empty masks
    ocr_book_spines = []
    for bookspine_mask in bookspine_masks:
        bookspine_mask_polygon = Polygon(bookspine_mask)
        bookspine_isolated_np = extract_mask_array_from_image(input_image, bookspine_mask_polygon)
        rotate_to_flat_angle = get_rotate_to_flat_angle(bookspine_mask_polygon)
        bookspine_isolated_rotated_to_flat_np = scipy_rotate(bookspine_isolated_np, rotate_to_flat_angle, reshape=True)
        ocr_results = get_ocr_results_all_rotations(bookspine_isolated_rotated_to_flat_np, reader)
        ngrams_dict = generate_ngrams_dict(ocr_results)
        ocr_results_coherence_scores = calculate_ocr_coherence_scores(ngrams_dict, ngram_data_df)
        most_coherent_text = get_most_coherent_text(ocr_results, ocr_results_coherence_scores)
        if len(most_coherent_text) > 0:  # no purpose in storing blank results if they somehow happen
            ocr_book_spines.append(most_coherent_text)

    book_info_from_query = [query_book_info_from_book_spine(_) for _ in ocr_book_spines]
    formatted_query_results = [f"{_['title']} by {', '.join(_['authors'])}" for _ in book_info_from_query]

    toc = time.perf_counter()
    print(f"application runs in {toc-tic:0.2f} seconds")
    return formatted_query_results


def get_rotate_to_flat_angle(mask_polygon: Polygon) -> float:
    centroid = mask_polygon.centroid.coords[0]

    Ix = 0
    Iy = 0
    Ixy = 0

    for i in range(len(mask_polygon.exterior.coords) - 1):
        xi, yi = mask_polygon.exterior.coords[i]
        xi1, yi1 = mask_polygon.exterior.coords[i + 1]

        Ai = xi * yi1 - xi1 * yi
        xi_avg = (xi + xi1) / 2
        yi_avg = (yi + yi1) / 2

        Ix += Ai * (yi_avg - centroid[1]) ** 2
        Iy += Ai * (xi_avg - centroid[0]) ** 2
        Ixy += Ai * (xi_avg - centroid[0]) * (yi_avg - centroid[1])

    theta_rad = 0.5 * np.arctan2(2 * Ixy, Ix - Iy)
    theta_deg = np.degrees(theta_rad)
    rotate_to_flat = -theta_deg
    return rotate_to_flat


def get_polygon_bounds_for_slicing(mask_polygon):
    bounds_int = [int(_) for _ in mask_polygon.bounds]  # .bounds returns floats, convert to int for slicing
    min_x, min_y, max_x, max_y = bounds_int
    return min_x, min_y, max_x, max_y


def extract_mask_array_from_image(input_image: Image, mask_polygon: Polygon) -> np.array:
    mask_bitmap = rasterio.features.rasterize([mask_polygon], out_shape=(input_image.height, input_image.width))
    input_image_bw = input_image.convert("L")
    input_image_masked = input_image_bw * mask_bitmap
    min_x, min_y, max_x, max_y = get_polygon_bounds_for_slicing(mask_polygon)
    input_image_masked_bounded_array = input_image_masked[min_y:max_y, min_x:max_x]

    return input_image_masked_bounded_array


def get_ocr_results_all_rotations(book_horizontal_array, reader) -> dict:
    angles = [0, 90, 180, 270]
    ocr_results = {}
    for angle in angles:
        rotated_array = scipy_rotate(book_horizontal_array, angle, reshape=True)
        bookspine_ocr = reader.readtext(rotated_array, detail=0)
        ocr_results[angle] = ' '.join([_.upper() for _ in bookspine_ocr])

    return ocr_results


def generate_ngrams_dict(ocr_results) -> dict:
    ngrams_dict = {}
    ngram_len = 3
    for angle, ocr_text_result in ocr_results.items():
        ngrams_ = []
        for word in ocr_text_result.split(' '):
            for start in range(0, len(word) - ngram_len - 1):
                ngram = word[start:start + ngram_len]
                ngrams_.append(ngram)

        ngrams_dict[angle] = ngrams_

    return ngrams_dict


def calculate_ocr_coherence_scores(ngrams_dict, ngram_data_df) -> dict:
    ocr_results_coherence_scores = {}
    for angle, ngrams in ngrams_dict.items():
        sum = 0
        for ngram in ngrams:
            df_lookup = ngram_data_df[ngram_data_df['3-gram'] == ngram]['*/*'].values
            if len(df_lookup) > 0:
                sum += df_lookup[0]
        ocr_results_coherence_scores[angle] = (
                sum / len(ngrams)) if ngrams else 0  # protect against divide by 0 error on empty lists
    return ocr_results_coherence_scores


def get_most_coherent_text(ocr_results, ocr_results_coherence_scores):
    most_coherent_angle = max(ocr_results_coherence_scores, key=ocr_results_coherence_scores.get)
    return ocr_results[most_coherent_angle]


def query_book_info_from_book_spine(book_spine_text: str) -> dict:
    query = {
        "q": book_spine_text,
        "maxResults": 1,
        "orderBy": "relevance"
    }
    response = requests.get('https://www.googleapis.com/books/v1/volumes/', params=query)
    response_json = response.json()
    # print(response_json)
    if response_json['totalItems'] > 0:
        book_info = response_json['items'][0]['volumeInfo']
    else:
        book_info = {'title': 'unable to find match', 'authors': ['unknown']}
    return book_info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
