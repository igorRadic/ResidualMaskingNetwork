import os
import warnings
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

import models
from utils.augmenters.augment import seg
from utils.generals import make_batch

# Desired camera to use.
CAMERA = 0
# Face detection mode: 0 - for closer face, 1 for away face.
DETECTION_MODEL = 0
# Default minimal face detection confidence.
MIN_DETECTION_CONFIDENCE = 0.7

# Don't print warnings.
warnings.filterwarnings("ignore")


class RMN_ensemble:
    def __init__(self):
        self._model_weights = [
            # Model architecture, weights name
            ("resnet101", "resnet101_rot30_2019Nov14_18.12"),
            ("cbam_resnet50", "cbam_resnet50_rot30_2019Nov15_12.40"),
            ("efficientnet_b2b", "efficientnet_b2b_rot30_2019Nov15_20.02"),
            ("resmasking_dropout1", "resmasking_dropout1_rot30_2019Nov17_14.33"),
            ("resmasking", "resmasking_rot30_2019Nov14_04.38"),
        ]
        self._models = []
        self._tta_size = 8
        self._image_size = (224, 224)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def load_model(self):
        for model_name, checkpoint_path in self._model_weights:

            print("Loading ", checkpoint_path)

            model = getattr(models, model_name)
            model = model(in_channels=3, num_classes=7)

            state = torch.load(os.path.join("saved/checkpoints", checkpoint_path))
            model.load_state_dict(state["net"])

            model.cuda()
            model.eval()

            self._models.append(model)

    def prepare_image_for_prediction(self, image):
        # Get gray image from bgr image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to suit model input size.
        image = cv2.resize(image, self._image_size)

        # Convert 1 channel image to 3 channel image.
        image = np.dstack([image] * 3)

        images = [seg(image=image) for i in range(self._tta_size)]
        # Apply transform to each element in the image list and
        # convert the result into a new list of images.
        images = list(map(self._transform, images))

        # Add batch size.
        images = make_batch(images)

        # Move images to CUDA-enabled GPU.
        images = images.cuda(non_blocking=True)

        return images

    def predict_emotion(self, images):

        prediction_list = []

        for model in self._models:
            with torch.no_grad():

                # Predict.
                output = model(images).cpu()

                # Perform Softmax.
                output = F.softmax(output, 1)

                # Sum predictions.
                output = torch.sum(output, 0)

                # Round to 4 decimal places.
                output = [round(o, 4) for o in output.numpy()]

                # Append model output to prediction list.
                prediction_list.append(output)

        # Convert list to numpy array
        prediction_list = np.array(prediction_list)

        # Sum elements on same place
        prediction_list = np.sum(prediction_list, axis=0)

        # Get predicted class (it is index of highest value in prediction list).
        current_prediction = np.argmax(prediction_list, axis=0)

        # Get confidence (it is average confidence of all models, rounded to 4 decimal places).
        classification_confidence = round(
            prediction_list[current_prediction] / len(self._models), 4
        )

        return current_prediction, classification_confidence


def realtime_facial_emotion_recognition():

    prediction_time_sum = 0
    predictions_counter = 0
    prev_frame_time = 0
    current_frame_time = 0
    classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Get desired camera as input, set font.
    cap = cv2.VideoCapture(CAMERA)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load classification model.
    model = RMN_ensemble()
    model.load_model()

    # Load detection model from Media Pipe.
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=DETECTION_MODEL,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    ) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Mark the image as not writeable to improve performance,
            # then it passes the image by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces on captured frame.
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, num_of_channels = image.shape

            # If face is detected, predict class for face on captured frame.
            if results.detections:
                try:
                    for detection in results.detections:
                        face_bounding_box = (
                            detection.location_data.relative_bounding_box
                        )

                        # Get x, y coordinates from top left corner of bounding box.
                        x, y = int(face_bounding_box.xmin * image_width), int(
                            face_bounding_box.ymin * image_height
                        )

                        # Get height and width of bounding box.
                        bBox_width = x + int(face_bounding_box.width * image_width)
                        bBox_height = y + int(face_bounding_box.height * image_height)

                        # Draw bounding box to the original image.
                        cv2.rectangle(
                            img=image,
                            pt1=(x, y),
                            pt2=(bBox_width, bBox_height),
                            color=(255, 255, 255),
                            thickness=2,
                        )

                        # Cropp bounding box.
                        face_image = image[y:bBox_height, x:bBox_width]

                        # Adjust face image to suit model and perform TTA method.
                        face_images = model.prepare_image_for_prediction(
                            image=face_image
                        )

                        # Predict emotion and measure prediction time.
                        start = perf_counter()
                        (
                            current_prediction,
                            classification_confidence,
                        ) = model.predict_emotion(images=face_images)
                        end = perf_counter()

                        # Don't measure first prediction time, it takes longer to predict.
                        if predictions_counter != 0:
                            prediction_time = end - start
                            prediction_time_sum += prediction_time

                            # Print average classification time.
                            cv2.putText(
                                img=image,
                                text=f"Avg class. time: {round(prediction_time_sum / predictions_counter, 3)} s",
                                org=(10, 50),
                                fontFace=font,
                                fontScale=0.5,
                                color=(100, 255, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA,
                            )
                        predictions_counter += 1

                        # Show prediction results if confidence is enough.
                        if classification_confidence > 0.65:
                            class_idx = current_prediction

                        # Print class.
                        cv2.putText(
                            img=image,
                            text=str(classes[class_idx]),
                            org=(x + 5, y - 5),
                            fontFace=font,
                            fontScale=0.5,
                            color=(255, 255, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                except:
                    pass

            # Calculate FPS.
            current_frame_time = perf_counter()
            fps = 1 / (current_frame_time - prev_frame_time)
            prev_frame_time = current_frame_time
            fps = int(fps)

            # Print FPS.
            cv2.putText(
                img=image,
                text=f"FPS: {fps}",
                org=(10, 20),
                fontFace=font,
                fontScale=0.5,
                color=(100, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Face emotion recognition", image)

            if cv2.waitKey(5) & 0xFF == 27:  # press 'ESC' to quit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_facial_emotion_recognition()
