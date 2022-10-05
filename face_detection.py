import cv2
import dlib
import numpy as np
from imutils import face_utils


def blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    mask = np.zeros(image.shape[:2], np.uint8)
    blurred_image = image.copy()
    for face in faces:  # if there are faces
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        shape = cv2.convexHull(shape)
        cv2.drawContours(mask, [shape], -1, 255, -1)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

    return mask


if __name__ == '__main__':

    # load the face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\37060\Documents\GitHub\magistras\deepfake-playground\shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(r"C:\Users\37060\Desktop\smth\data\001\001_01_01_041_06_crop_128.png")
    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect the faces
    rects = detector(gray)

    mask = np.zeros(image.shape[:2], np.uint8)

    # go through the face bounding boxes
    for rect in rects:
        # extract the coordinates of the bounding box
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # apply the shape predictor to the face ROI
        predicted_shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(predicted_shape)
        shape = cv2.convexHull(shape)
        cv2.drawContours(mask, [shape], -1, 255, -1)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        # get_face_mask(image, )

        for n in range(0, 68):
            x = predicted_shape.part(n).x
            y = predicted_shape.part(n).y
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)


    # mask = blur(image)
    cv2.imwrite('color_img.jpg', mask)
    cv2.imshow("image", mask)
    cv2.waitKey()
    cv2.imshow("Image", image)
    cv2.waitKey(0)

