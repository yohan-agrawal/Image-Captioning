import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add
from tensorflow.keras.utils import to_categorical
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Load and Preprocess Captions
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:  # Open the file in read mode
        for line in file:
            tokens = line.strip().split(",", 1)  # Split on the first comma only
            if len(tokens) != 2:
                continue
            image_id, caption = tokens[0].split('.')[0], tokens[1].lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(f"<start> {caption} <end>")
    return captions

# Example usage: Load captions
captions_file_path = "C:\\AI PROJECTS\\Image Captioning\\captions.txt"  # Update path if needed
captions_dict = load_captions(captions_file_path)

# Step 2: Initialize Tokenizer
all_captions = [caption for captions in captions_dict.values() for caption in captions]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_caption_len = max(len(caption.split()) for caption in all_captions)

# Step 3: Extract Image Features Using VGG16
def extract_features(image_path):
    model = VGG16(weights='imagenet', include_top=False)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features

# Define the folder where all images are stored
image_folder = "C:\\AI PROJECTS\\Image Captioning\\Images"  # Update this path to your image dataset folder

# Initialize a dictionary to hold features for each image
features_dict = {}
for image_name in os.listdir(image_folder):
    image_id = image_name.split('.')[0]
    image_path = os.path.join(image_folder, image_name)
    features_dict[image_id] = extract_features(image_path)

print(f"Loaded features for {len(features_dict)} images.")

# Step 4: Preprocess Captions for Training
def encode_captions(captions):
    sequences = tokenizer.texts_to_sequences(captions)
    return pad_sequences(sequences, maxlen=max_caption_len, padding="post")

encoded_captions = {img: encode_captions(captions) for img, captions in captions_dict.items()}

# Step 5: Define the Image Captioning Model
def define_model(vocab_size, max_length):
    # Image feature extractor
    inputs1 = Input(shape=(7, 7, 512))  
    fe1 = Dense(256, activation="relu")(inputs1)
    fe2 = tf.keras.layers.Flatten()(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    
    # Decoder model
    decoder1 = add([fe2, se2])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

model = define_model(vocab_size, max_caption_len)
model.summary()

# Step 6: Training the Model (Simplified Example with Mock Data)
# Generate training examples
def data_generator(captions_dict, features_dict, tokenizer, max_length, vocab_size):
    while True:
        for img_id, captions in captions_dict.items():
            image_features = features_dict[img_id][0]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [[image_features, in_seq], out_seq]

# Uncomment below line to train the model
# model.fit(data_generator(captions_dict, features_dict, tokenizer, max_caption_len, vocab_size),
#           steps_per_epoch=len(captions_dict), epochs=20, verbose=1)

# Step 7: Generate Caption for a New Image
def generate_caption(model, tokenizer, photo, max_len):
    input_text = "<start>"
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="post")
        yhat = model.predict([photo, sequence], verbose=0)
        word = tokenizer.index_word[np.argmax(yhat)]
        if word == "end":
            break
        input_text += " " + word
    return input_text.replace("<start>", "").strip()

# Example of generating a caption
# new_image_path = r"C:\AI PROJECTS\Image Captioning\images\example.jpg"  # Update to your test image
# photo = extract_features(new_image_path)
# caption = generate_caption(model, tokenizer, photo, max_caption_len)
# print("Generated Caption:", caption)