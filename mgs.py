#!/usr/bin/env python
from os import path
from scipy import ndimage
import cv2
import numpy as np
import sys

lorem = """Ipsum consequuntur temporibus veniam ut. Incidunt eveniet pariatur illum qui. Reprehenderit perspiciatis
repudiandae voluptatum nobis rerum deleniti ut reiciendis. Incidunt quae doloribus accusantium eos id ducimus unde
omnis. Eum tempore doloremque amet deserunt similique.
Maxime sit amet sint. Voluptas totam ullam voluptatibus qui. Quae nihil veritatis ipsam. In quisquam qui iusto qui illum
iure. Ab libero quia rerum ea nostrum consectetur. Blanditiis cum ex rerum eum similique beatae.
Dolores hic provident nesciunt quia repellat aut ducimus. Temporibus delectus accusantium aut. Sequi repudiandae a eum
quam officiis.
Porro non vel in. Voluptatem ducimus culpa aut hic pariatur sit quis quam. Et fuga autem enim. Eos nemo natus assumenda.
Et tempora qui numquam aliquid at.
Temporibus dolor enim sint ut earum dolor velit quis. Cum sit ratione similique quia unde. Dolores veritatis iure
asperiores facere debitis natus. Consequuntur eum corrupti eum necessitatibus eveniet distinctio et omnis. Vero veniam
excepturi incidunt."""

def get_eyes(image):
    """Get list of eyes (x, y, w, h) boundaries for eye pairs."""

    eyePairs = []

    gray = grey(image)

    face_cascade = cv2.CascadeClassifier(path.join(path.dirname(__file__), 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(path.join(path.dirname(__file__), 'haarcascade_eye.xml'))

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (fx, fy, fw, fh) in faces:
        gray_face = gray[fy : fy + fh, fx : fx + fw]

        eyes = eye_cascade.detectMultiScale(gray_face)

        # expand the eye pairs to the largest boundary
        (minX, minY, maxX, maxY) = (sys.maxint, sys.maxint, 0, 0)
        for (ex, ey, ew, eh) in eyes:
            if ex < minX:
                minX = ex
            if ey < minY:
                minY = ey
            if ex+ew > maxX:
                maxX = ex + ew
            if ey+ew > maxY:
                maxY = ey + eh

        # expand the eyes
        (x1, x2) = (minX + fx, maxX + fx)
        (x1, x2) = (int(x1 * 0.8), min(int(x2 * 1.2), image.shape[1]))

        eyePairs.append((x1, minY + fy, x2, maxY + fy))
    return eyePairs

def contrast(image, maxIntensity=100, phi=1, theta=1):
    """Increase contrast."""
    image = image * 0.75
    image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 0.5
    return cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def normalize(image):
    """Normalize image after adding noise or other values."""
    return cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def median_filter(image, blur):
    """Apply a combination of random noise and bluring."""
    noisy = image + image.std() * np.random.random(image.shape)
    noisy = ndimage.median_filter(noisy, blur)
    return normalize(noisy)

def grey(image):
    """Convert color image to grey."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def color(image):
    """Convert grey image to color."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def sepia(image):
    """Convert image to sepia."""
    sepia_transform = np.asarray([[0.393, 0.769, 0.189],
                                  [0.349, 0.686, 0.168],
                                  [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, sepia_transform)
    # I think this is just a color transformation to account for sepia
    sepia_image = cv2.cvtColor(sepia_image, cv2.cv.CV_RGB2BGR)
    return sepia_image

def face_noise_filter(image):
    """Make the picture noisy and sepia."""
    noisy = contrast(image)
    noisy = median_filter(noisy, 3)
    noisy = median_filter(noisy, 2)
    return sepia(color(grey(noisy)))

def write_text(image, text, x, y, size=1, thickness=3, color=(0, 0, 255)):
    """Write text on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x, y), font, size, color, thickness, bottomLeftOrigin=True)

    return image

def censor_eyes(image, eyePairs):
    """
    MGS eye censor effect does the following:
    - One box on the left (about 1/3) and a box on the right
    - A number is in the upper center of the left box
    - All color within the box is tinted red
    """
    for (x1, y1, x2, y2) in eyePairs:
        w = x2 - x1
        h = y2 - y1
        xm = w / 3 + x1
        cv2.rectangle(image, (x1, y1), (xm, y2), (0, 0, 255), 3)
        cv2.rectangle(image, (xm, y1), (x2, y2), (0, 0, 255), 3)
        textx = xm + x1
        image = write_text(image, "28", textx/2, y1)

        image_slice = image[y1 : y2, x1 : x2]
        for ix, row in enumerate(image_slice):
            for iy, color in enumerate(row):
                (b, g, r) = color
                image_slice[ix][iy] = (0, 0, r)

        w2 = w - w / 3
        text_image = np.zeros((125, 125 * w2 / h, 3), np.uint8)
        for (i, line) in enumerate(lorem.splitlines()):
            text_image = write_text(text_image, line, 0, i * 10, 0.25, 1)
        text_image = cv2.resize(text_image, (w2, h))

        image_subslice = image_slice[0:h,w/3:w]
        for i, row in enumerate(image_subslice):
            image_subslice[i] = row + text_image[i] / 255.

    return image

def profile_text(image):
    """
    Add red and white text at bottom of the image about 4/5ths the screen.
    """
    (h, w, _) = image.shape
    (x1, y1, x2, y2) = (0, h - h / 6, w * 5 / 6, h)
    (tw, th) = (x2 - x1, y2 - y1)

    text_image = np.zeros((125, 125 * tw / th, 3), np.uint8)
    for (i, line) in enumerate(lorem.splitlines()):
        text_image = write_text(text_image, line, 0, i * 10, 0.25, 1)
    text_image = cv2.resize(text_image, (tw, th))


    image_slice = image[y1:y2,x1:x2]
    for i, row in enumerate(image_slice):
        image_slice[i] = row + text_image[i] / 255.

    return image

def show(image):
    """
    Show the image.
    """
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    """
    Perform the filter.
    """

    # NOTE: I know there are some misconceptions in this code about the way images are handled by cv2. Specifically,
    # normalizing has been guess and check. I also think I might be transposing x and y coordinates in this code in
    # places. That said, the end result gives the effect I want, so I haven't corrected it.
    face = cv2.imread(args.filename)

    face = cv2.resize(face, (600, 600 * face.shape[0] / face.shape[1]))

    eyePairs = get_eyes(face)
    face = face_noise_filter(face)
    face = censor_eyes(face, eyePairs)
    face = profile_text(face)

    # add some more noise
    face2 = median_filter(face, 10)
    face = (face + face2) / 2

    # shrink the image
    face = cv2.resize(face, (args.width, args.width * face.shape[0] / face.shape[1]))

    if args.output == None:
        show(face)
    else:
        face = face * 255

        cv2.imwrite(args.output, face)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Apply MGS filter to profile picture.')
    parser.add_argument('filename', help='Input filename')
    parser.add_argument('--width', dest='width', type=int, default=100, help='Output image width')
    parser.add_argument('--output', dest='output', help='Output filename')
    args = parser.parse_args()
    main(args)
