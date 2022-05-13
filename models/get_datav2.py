import glob
from keras.preprocessing.image import ImageDataGenerator


def gather_images(dir="../static/image_db/"):
    train = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test = ImageDataGenerator()
    train_set = train.flow_from_directory(
        dir, target_size=(64, 64), batch_size=32, class_mode="categorical", color_mode="grayscale"
    )
    test_set = test.flow_from_directory(
        dir, target_size=(64, 64), batch_size=32, class_mode="categorical", color_mode="grayscale"
    )
    faces = {}
    for faceValue, faceName in zip(train_set.class_indices.values(),
            train_set.class_indices.keys()):
        faces[faceValue] = faceName
    return train_set, test_set, faces
