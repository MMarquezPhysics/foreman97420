from tensorflow.io import decode_jpeg
from tensorflow.image import resize
from tensorflow import expand_dims

CLASSES = ["Glioma", 'Meningioma', 'Ausencia de Tumor (Sano)', 'Pituitario']
TARGET_SIZE = (224, 224)

def preprocess_uploaded_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    image = decode_jpeg(file_bytes, channels=3)
    image = resize(image, TARGET_SIZE)
    image = image / 255.0
    image = expand_dims(image, axis=0)
    return image

def predict_tumor_probabilities(model, preprocessed_tensor):
    predictions = model.predict(preprocessed_tensor, verbose=0)[0]
    probability_dict = {
        CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
    return probability_dict