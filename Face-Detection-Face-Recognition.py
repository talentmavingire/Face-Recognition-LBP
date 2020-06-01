# Import core packages
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


st.title("Face Detection and Face Recognition Using Neural Networks")
# load image
image_file = st.file_uploader(
    "Upload Image", type=['jpg', 'png', 'jpeg'])


@st.cache
def load_image(img):
    image = Image.open(img)
    return image


# cascade classifier
faceCascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# check for image upload
if image_file is not None:
    st.text("Original Image")
    upload_image = load_image(image_file)
    st.image(upload_image, 300, 300)


def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(20, 20)
    )
    # Draw rectangle bounding box around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, faces


def enhance_image(image):
    enhance_type = st.sidebar.radio(
        "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Blurring"])
    if enhance_type == 'Gray-Scale':
        new_img = np.array(upload_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.write(enhance_type)
        st.image(gray, width=300)

    elif enhance_type == 'Contrast':
        c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        enhancer = ImageEnhance.Contrast(upload_image)
        img_output = enhancer.enhance(c_rate)
        st.write(enhance_type)
        st.image(img_output, width=300)

    elif enhance_type == 'Blurring':
        new_img = np.array(upload_image.convert('RGB'))
        blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        img = cv2.cvtColor(new_img, 1)
        blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
        st.write(enhance_type)
        st.image(blur_img, width=300)

    elif enhance_type == 'Original':
        st.image(upload_image, width=300)
    else:
        st.image(upload_image, width=300)


def main():

    activities = ["", "Face Detection",
                  "Enhance Image", "Face Recogntion", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Face Detection' or 'Face Recognition':
        st.subheader(choice)

        # Face Detection
        task = ["Detection", "Recognition"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):

            if feature_choice == 'Detection':
                result_img, result_faces = detect_faces(upload_image)
                st.image(result_img, 400, 400)

                st.success("Found {} faces".format(len(result_faces)))
            # Face Recognition
            elif feature_choice == 'Recognition':

                # check confidence
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read('dataset/trainer.yml')
                cascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(cascadePath)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # iniciate id counter
                id = 0
                # names related to ids: example ==> Paul: id=1,  etc
                names = ['None', 'Paul', 'Paul', 'Paul', 'Paul',
                         'Paul', 'Conrad', "Conrad", 'Conrad', "Conrad", 'Conrad']
                # face
                result_img, result_faces = detect_faces(upload_image)
                for (x, y, w, h) in result_faces:
                    # draw bounding box
                    cv2.rectangle(result_img, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)
                    new_image = np.array(upload_image.convert('RGB'))
                    image = cv2.cvtColor(new_image, 1)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    id, confidence = recognizer.predict(
                        gray[y: y + h, x: x + w])

                    # check cofidence
                    if (confidence < 100):
                        id = names[id]
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        id = "unknown"
                        confidence = "  {0}%".format(round(100 - confidence))
                        cv2.putText(result_img, str(id),
                                    (x + 5, y - 5), font, 1, (0, 255, 0), 2)
                        cv2.putText(result_img, str(confidence), (x + 5, y + h - 5),
                                    font, 1, (255, 255, 0), 1)
                st.image(result_img, 400, 400)

    elif choice == "Enhance Image":
        enhance_image(upload_image)

    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown(
            "Built with Streamlit by [Paul](https://github.com/talentmavingire/)")
        st.text("Talent Paul Mavingire")


if __name__ == '__main__':
    main()
