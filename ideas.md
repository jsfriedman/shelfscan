entire idea:
* user opens app on phone
* live view of camera
* point at bookshelf
* visual indicator of which books the user might enjoy

potential starting point:
* single image as input
* identify all books spines in the image
* determine text on the bookspine
* resolve text to a Book
* see how that book scores on the users profile
* return set of recommended books


less prescriptive outline:
* image (end result is live camera stream on mobile device. static images for now)
* books (idea is to OCR based on a book spine region, but maybe just OCR)
* recommendation (need to find a service like Goodreads with an API)


SAM ViT-L weights (git ignored): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

cd weights
curl.exe -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth