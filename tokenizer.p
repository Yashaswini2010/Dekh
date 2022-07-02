{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91b8aa0a-aba3-4d0d-9c8c-ba5d54b2f63f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def dict_to_list(descriptions):\n",
    "   all_desc = []\n",
    "for key in descriptions.keys():\n",
    "       [all_desc.append(d) for d in descriptions[key]]\n",
    "return all_desc\n",
    "#creating tokenizer class\n",
    "#this will vectorise text corpus\n",
    "#each integer will represent token in dictionary\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "def create_tokenizer(descriptions):\n",
    "   desc_list = dict_to_list(descriptions)\n",
    "   tokenizer = Tokenizer()\n",
    "   tokenizer.fit_on_texts(desc_list)\n",
    "return tokenizer\n",
    "# give each word an index, and store that into tokenizer.p pickle file\n",
    "tokenizer = create_tokenizer(train_descriptions)\n",
    "dump(tokenizer, open('tokenizer.p', 'wb'))\n",
    "vocab_size = len(tokenizer.word_index) + 1\n",
    "Vocab_size #The size of our vocabulary is 7577 words.\n",
    "#calculate maximum length of descriptions to decide the model structure parameters.\n",
    "def max_length(descriptions):\n",
    "   desc_list = dict_to_list(descriptions)\n",
    "return max(len(d.split()) for d in desc_list)\n",
    "max_length = max_length(descriptions)\n",
    "Max_length"
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
