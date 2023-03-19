import os
import tensorflow as tf
import cv2 as cv
import time

def capture_image():
    
    capture = cv.VideoCapture(0)
    counter = 0
    while True:

        isTrue, frame = capture.read()
        # print(frame)
        cv.imwrite('./images/{}.png'.format(counter), frame)
        cv.imshow('Video', frame)
        time.sleep(1)
        counter +=1
        if (cv.waitKey(20) & 0xFF == ord('d')):

            break

    capture.release()
    cv.destroyAllWindows()


train_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/deeplearning/train",
        validation_split = 0.2,
        subset = "training",
        #color_mode = "grayscale",
        seed = 123,
        image_size = (200, 200),
        batch_size = 32
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/deeplearning/test",
        validation_split = 0.2,
        subset = "validation",
        #color_mode = "grayscale",
        seed = 123,
        image_size = (200, 200),
        batch_size = 32
    )

model = tf.keras.Sequential([
        # Data augmentation for better recognition

        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        # tf.image.central_crop(0.5),

        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(3, activation = "softmax")
    ])

model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
model.fit(train_ds, validation_data = val_ds, epochs = 20)
model.save(os.path.join(ml_model_directory, "model.h5"))
print("[green]Model saved successfully[/green]") 




def predict_frame(model, frame):

    """
    Function to predict the class of frame captured by open-cv.
    
    ---
    
    Args:
    `model`: The trained machine learning model
    `frame`: Frame captured by open-cv
    ---
    Return:
    `Status`: True/ False depending on whether anybody was identified with a certain level of accuracy.
    `User ID`: If the status is True, then function will also return the User ID of the identified user in the frame. Else it'll return None
    """
    successive_feature_maps = model.predict(frame)

    # Get ID's of users in order of successive_feature_maps
    classes=["red","blue","yellow"]
    try:
        predicted_user_ids = [name for name in classes]
    except Exception as e:
        print("[red]Fetching Predicted User ID Failed[/red]: [yellow]There might not be any users who have registered their face. Hence kindly make them register face to start FRAS.[/yellow]")
        return [False, False]

    max_predicted_probability = max(successive_feature_maps[0])

    # Confirm a frame to be of a user only if predicted probability comes out to be > 60%
    print("Max probability is: ",max_predicted_probability, successive_feature_maps, predicted_user_ids)
    if(max_predicted_probability <= 0.60):
        return [False, None]

    else:
        return [True, predicted_user_ids[list(successive_feature_maps[0]).index(max_predicted_probability)]]

def identify_image():

    pass

def recognize_face():

    capture = cv.VideoCapture(0)
    try:
        model = tf.keras.models.load_model(os.path.join(ml_model_directory, "model.h5"))
    except Exception as e:
        print("[yellow]Unable to load ML model[/yellow]/nTrying to create a new model...")
        face_recognition_model()
        model = tf.keras.models.load_model(os.path.join(ml_model_directory, "model.h5"))

    while True:

        isTrue, frame = capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (200, 200))
        frame = np.array(frame)
        frame = frame.reshape((1, )+ frame.shape)

        prediction = predict_frame(model, frame)
        print(prediction)
        time.sleep(1)

        if (cv.waitKey(20) & 0xFF == ord('d')):
            print("[yellow]Aborting camera recording[/yellow]")
            break
        