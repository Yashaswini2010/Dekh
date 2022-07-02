import numpy as np
import gzip, pickle
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm #to check loop progress
tqdm().pandas()
def load_fp(dekh):
  # Open file to read
   file = open(dekh, 'r')
   text = file.read()
   file.close()
return text
# get all images with their captions
def img_capt(dekh):
   file = load_doc(dekh)
   captions = file.split('n')
   descriptions ={}
for caption in captions[:-1]:
       img, caption = caption.split('t')
if img[:-2] not in descriptions:
           descriptions[img[:-2]] = [ caption ]
else:
           descriptions[img[:-2]].append(caption)
return descriptions
#Data cleaning function will convert all upper case alphabets to lowercase, removing punctuations and words containing numbers
def txt_clean(captions):
   table = str.maketrans('','',string.punctuation)
for img,caps in captions.items():
for i,img_caption in enumerate(caps):
           img_caption.replace("-"," ")
           descp = img_caption.split()
          #uppercase to lowercase
           descp = [wrd.lower() for wrd in descp]
          #remove punctuation from each token
           descp = [wrd.translate(table) for wrd in descp]
          #remove hanging 's and a
           descp = [wrd for wrd in descp if(len(wrd)>1)]
          #remove words containing numbers with them
           descp = [wrd for wrd in descp if(wrd.isalpha())]
          #converting back to string
           img_caption = ' '.join(desc)
           captions[img][i]= img_caption
return captions
def txt_vocab(descriptions):
  # To build vocab of all unique words
   vocab = set()
for key in descriptions.keys():
       [vocab.update(d.split()) for d in descriptions[key]]
return vocab
#To save all descriptions in one file
def save_descriptions(descriptions, dekh):
   lines = list()
for key, desc_list in descriptions.items():
for desc in desc_list:
           lines.append(key + 't' + desc )
   data = "n".join(lines)
   file = open(dekh,"w")
   file.write(data)
   file.close()
# Set these path according to project folder in you system, like i create a folder with my name shikha inside D-drive
dataset_text = "D:shikhaProject - Image Caption GeneratorFlickr_8k_text"
dataset_images = "D:shikhaProject - Image Caption GeneratorFlicker8k_Dataset"
#to prepare our text data
dekh = dataset_text + "/" + "Flickr8k.token.txt"
#loading the file that contains all data
#map them into descriptions dictionary 
descriptions = img_capt(dekh)
print("Length of descriptions =" ,len(descriptions))
#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving all descriptions in one file
save_descriptions(clean_descriptions, "descriptions.txt")
model = Xception( include_top=False, pooling='avg' )
def load_photos(dekh):
   file = load_doc(dekh)
   photos = file.split("n")[:-1]
return photos
def load_clean_descriptions(dekh, photos):
  #loading clean_descriptions
   file = load_doc(dekh)
   descriptions = {}
for line in file.split("n"):
       words = line.split()
if len(words)<1 :
           continue
       image, image_caption = words[0], words[1:]
if image in photos:
if image not in descriptions:
               descriptions[image] = []
           desc = ' ' + " ".join(image_caption) + ' '
           descriptions[image].append(desc)
return descriptions
def load_features(photos):
  #loading all features
   all_features = pickle.load(open("/home/naruto/dekh/features.pklz","rb"))
  #selecting only needed features
   features = {k:all_features[k] for k in photos}
return features
dekh = dataset_text + "/" + "Flickr_8k.trainImages.txt"
#train = loading_data(filename)
train_imgs = load_photos(dekh)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)
def data_generator(descriptions, features, tokenizer, max_length):
while 1:
for key, description_list in descriptions.items():
          #retrieve photo features
           feature = features[key][0]
           inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature)
           yield [[inp_image, inp_sequence], op_word]
def create_sequences(tokenizer, max_length, desc_list, feature):
   x_1, x_2, y = list(), list(), list()
  # move through each description for the image
for desc in desc_list:
      # encode the sequence
       seq = tokenizer.texts_to_sequences([desc])[0]
      # divide one sequence into various X,y pairs
for i in range(1, len(seq)):
          # divide into input and output pair
           in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
           in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
           out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
           x_1.append(feature)
           x_2.append(in_seq)
           y.append(out_seq)
return np.array(X_1), np.array(X_2), np.array(y)
#To check the shape of the input and output for your model
[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape
from keras.utils import plot_model
# define the captioning model
def define_model(vocab_size, max_length):
  # features from the CNN model compressed from 2048 to 256 nodes
   inputs1 = Input(shape=(2048,))
   fe1 = Dropout(0.5)(inputs1)
   fe2 = Dense(256, activation='relu')(fe1)
  # LSTM sequence model
   inputs2 = Input(shape=(max_length,))
   se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
   se2 = Dropout(0.5)(se1)
   se3 = LSTM(256)(se2)
  # Merging both models
   decoder1 = add([fe2, se3])
   decoder2 = Dense(256, activation='relu')(decoder1)
   outputs = Dense(vocab_size, activation='softmax')(decoder2)
  # merge it [image, seq] [word]
   model = Model(inputs=[inputs1, inputs2], outputs=outputs)
   model.compile(loss='categorical_crossentropy', optimizer='adam')
  # summarize model
   print(model.summary())
   plot_model(model, to_file='model.png', show_shapes=True)
return model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
# creating a directory named models to save our models
os.mkdir("models")
for i in range(epochs):
   generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
   model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
   model.save("models/model_" + str(i) + ".h5")
