# #!/usr/bin/env python3
# """
# Test AprilTag detection on the example tag image
# """

# import cv2
# import numpy as np
# from pupil_apriltags import Detector


# imagepath = "example_tag.png"
# image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# assert image is not None, "Failed to load image"

# # Normalize the image if it is not uint8
# # if image.dtype != "uint8":
# #    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

# # View image
# # cv2.imshow("Image", image)
# # cv2.waitKey(0) # Wait for a key press
# # cv2.destroyAllWindows()

# detector = Detector(
#     families="tagCircle21h7",
#     quad_decimate=1.0,
#     # quad_sigma=0.0,
#     # refine_edges=1,
#     # decode_sharpening=0.25
# )

# detections = detector.detect(image)
# print(detections)
