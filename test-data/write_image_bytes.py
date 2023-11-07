import io

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import imghdr

# # write image to byets
# image = Image.open("book_shelf3.jpg")
# with io.BytesIO() as output:
#     image.save(output, format='jpeg')
#     serialized_image = output.getvalue()
#
# with open("serialized_book_shelf3", "wb") as file:
#     file.write(serialized_image)


# read image from bytes
with open("serialized_book_shelf3", "rb") as file:
    serialized_image_bytes = file.read()

image_file_type = imghdr.what(None, h=serialized_image_bytes)
input_image = Image.open(io.BytesIO(serialized_image_bytes), formats=[image_file_type])
input_image.show()

# these lines work
# image_np_1d = np.frombuffer(serialized_image_bytes, np.uint8)
# img_np_2d = cv2.imdecode(image_np_1d, cv2.IMREAD_COLOR)
# cv_image_rgb = cv2.cvtColor(img_np_2d, cv2.COLOR_BGR2RGB)
# image = Image.fromarray(cv_image_rgb)


#
# deserialized_image = Image.open(io.BytesIO(serialized_image_bytes))
# deserialized_image.show()