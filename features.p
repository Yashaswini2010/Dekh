{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1150e076-3be3-4753-bc25-36abc0233b7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_features(directory):\n",
    "       model = Xception( include_top=False, pooling='avg' )\n",
    "       features = {}\n",
    "for pic in tqdm(os.listdir(dirc)):\n",
    "           file = dirc + \"/\" + pic\n",
    "           image = Image.open(file)\n",
    "           image = image.resize((299,299))\n",
    "           image = np.expand_dims(image, axis=0)\n",
    "          #image = preprocess_input(image)\n",
    "           image = image/127.5\n",
    "           image = image - 1.0\n",
    "           feature = model.predict(image)\n",
    "           features[img] = feature\n",
    "return features\n",
    "#2048 feature vector\n",
    "features = extract_features(dataset_images)\n",
    "dump(features, open(\"features.p\",\"wb\"))\n",
    "#to directly load the features from the pickle file.\n",
    "features = load(open(\"features.p\",\"rb\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
