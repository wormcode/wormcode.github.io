# -*- coding: utf-8 -*-
#!/usr/bin/python

# python3

 
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
 
import nltk
from nltk.tokenize import word_tokenize
"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ] 
"""
from nltk.stem import WordNetLemmatizer
"""
Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster.
"""
 
# http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz 
# download this file and rename the contained file with pos.txt and neg.txt respectively
pos_file = 'pos.txt'
neg_file = 'neg.txt'
 
# build word vocabulary table
def create_lexicon(pos_file, neg_file):
	lex = []
	
	def process_file(f):
		with open(pos_file, 'r') as f:
			lex = []
			lines = f.readlines()
			#print(lines)
			for line in lines:
				words = word_tokenize(line.lower())
				lex += words
			return lex
 
	lex += process_file(pos_file)
	lex += process_file(neg_file)
	#print(len(lex))
	lemmatizer = WordNetLemmatizer()
	lex = [lemmatizer.lemmatize(word) for word in lex] #  (cats->cat)
 
	word_count = Counter(lex)
	#print(word_count)
	# {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
	# remove some stop words like: the,a and  and so on，and some uncommon words;those words have nothing to do with the classification for positive or negative
	lex = []
	for word in word_count:
		if word_count[word] < 2000 and word_count[word] > 20:  # may use the  percentage
			lex.append(word)        # zipf law
	return lex
 
lex = create_lexicon(pos_file, neg_file)
#lex  reserve the words shows in the file。
 
# change each comment item to word vector, transfer rules：
# let lex is ['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] , in actual circumstance it is much bigger in length
# comment 'i think this movie is great' change to [0,1,0,0,0,0,0,1], words appears both in comments and Lex,mark the correspond index value as 1
# otherwise 0
def normalize_dataset(lex):
	dataset = []
	# lex:word vocabulary；review:comments；clf:classification of comment，[0,1] as negative comment, [1,0]as positive comment 
	def string_to_vector(lex, review, clf):
		words = word_tokenize(line.lower())
		lemmatizer = WordNetLemmatizer()
		words = [lemmatizer.lemmatize(word) for word in words]
 
		features = np.zeros(len(lex))
		for word in words:
			if word in lex:
				features[lex.index(word)] = 1  # word may appears many times in one sentence,you could use +=1，may make no difference
		return [features, clf]
 
	with open(pos_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [1,0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
			dataset.append(one_sample)
	with open(neg_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [0,1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
			dataset.append(one_sample)
	
	#print(len(dataset))
	return dataset
 
dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""
# persistence in file
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""
 
# use 10% samples  as test data
test_size = int(len(dataset) * 0.1)
 
dataset = np.array(dataset)
 
train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]
 
# Feed-Forward Neural Network
# define how many neurons each layer
n_input_layer = len(lex)  # input layer
 
n_layer_1 = 1000    # hide layer
n_layer_2 = 1000    # hide layer
 
n_output_layer = 2       # output layer
 
# define the Nueral network
def neural_network(data):
	# define weigths and biases of 1st layer
	layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
	# define weigths and biases of 2st layer
	layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
	# define weigths and biases of output layer
	layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}
 
	# w·x+b
	layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
	layer_1 = tf.nn.relu(layer_1)  # activation 
	layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
	layer_2 = tf.nn.relu(layer_2 ) # activation
	layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
 
	return layer_output
 
# 50 items in one batch
batch_size = 50
 
X = tf.placeholder('float', [None, len(train_dataset[0][0])]) 
#[None, len(train_x)] as the width and heigth (matrix) 
Y = tf.placeholder('float')

# training
def train_neural_network(X, Y):
	predict = neural_network(X)
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate default  0.001 
 
	epochs = 13
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		epoch_loss = 0
 
		i = 0
		random.shuffle(train_dataset)
		train_x = dataset[:, 0]
		train_y = dataset[:, 1]
		for epoch in range(epochs):
			while i < len(train_x):
				start = i
				end = i + batch_size
 
				batch_x = train_x[start:end]
				batch_y = train_y[start:end]
 
				_, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x),Y:list(batch_y)})
				epoch_loss += c
				i += batch_size
 
			print(epoch, ' : ', epoch_loss)
 
		text_x = test_dataset[: ,0]
		text_y = test_dataset[:, 1]
		correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accurate: ', accuracy.eval({X:list(text_x) , Y:list(text_y)}))
 
train_neural_network(X,Y)
