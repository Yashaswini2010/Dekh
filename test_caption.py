import numpy as np
import gzip, pickle
from PIL import Image
import matplotlib.pyplot as plt
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
def extract_features(dekh, model):
	try:
    			image = Image.open('/Flicker8k_Dataset/12.jpg')
	except:
    			print("ERROR: Can't open image! Ensure that image path and extension is correct")
	image = image.resize((299,299))
	image = np.array(image)
  # for 4 channels images, we need to convert them into 3 channels
	if image.shape[2] == 4:
		   image = image[..., :3]
	image = np.expand_dims(image, axis=0)
	image = image/127.5
	image = image - 1.0
	feature = model.predict(image)
	return feature
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
			if index == integer:
				return word
	return None
def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'start'
	for i in range(max_length):
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			sequence = pad_sequences([sequence], maxlen=max_length)
			pred = model.predict([photo,sequence], verbose=0)
			pred = np.argmax(pred)
			word = word_for_id(pred, tokenizer)
			if word is None:
					break
			in_text += ' ' + word
			if word == 'end':
				break
	return in_text
max_length = 32
tokenizer = pickle.load(open("/home/naruto/dekh/tokenizer.pklz","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)
with open('abc1.txt', mode='w') as file:
                file.write(description)

with open('abc1.txt','r') as file:
    info = file.read().rstrip('\n')

engine = pyttsx3.init()
engine.say(info)
engine.runAndWait()
