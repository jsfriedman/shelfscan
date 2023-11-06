import io
from PIL import Image

# write image to byets
image = Image.open("book_shelf3.jpg")
with io.BytesIO() as output:
    image.save(output, format='jpeg')
    serialized_image = output.getvalue()

with open("serialized_book_shelf3", "wb") as file:
    file.write(serialized_image)


# read image from bytes
# with open("serialized_book_shelf3", "rb") as file:
#     serialized_image_bytes = file.read()
#
# deserialized_image = Image.open(io.BytesIO(serialized_image_bytes))
# deserialized_image.show()