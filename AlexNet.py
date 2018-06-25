# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, initializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

# モデル設定
batch_size = 100  # バッチサイズ
n_epoch = 20  # エポック数
n_channel = 1  # channel数（画像の奥行的な。カラー画像ならRGBなので3、モノクロなら1）
n_label = 10  # 正解ラベルの種類数

class MLP(Chain):
	# 多層パーセプトロンによる分類
	def __init__(self):
		super(MLP, self).__init__()
		with self.init_scope():
			self.fc1 = L.Linear(None, 100)
			self.fc2 = L.Linear(100, n_label)
			self.bn1 = L.BatchNormalization(100)

	def __call__(self, x):
		h = F.sigmoid(self.fc1(x))
		h = self.bn1(h)
		return self.fc2(h)

class LeNet(Chain):
	# CNNを用いた分類
	def __init__(self):
		super(LeNet, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(n_channel,6,5,1)
			self.conv2 = L.Convolution2D(6,16,5,1)
			self.conv3 = L.Convolution2D(16,120,4,1)
			self.fc4 = L.Linear(None, 84)
			self.fc5 = L.Linear(84,n_label)

	def __call__(self, x):
		h = F.sigmoid(self.conv1(x))
		h = F.max_pooling_2d(h, 2, 2)
		h = F.sigmoid(self.conv2(h))
		h = F.max_pooling_2d(h,2,2)
		h = F.sigmoid(self.conv3(h))
		h = F.sigmoid(self.fc4(h))
		return self.fc5(h)

class Alex(Chain):
	# AlexNet
	def __init__(self):
	    super(Alex, self).__init__(
	        conv1 = L.Convolution2D(n_channel, 96, 11, stride=4),
	        conv2 = L.Convolution2D(96, 256, 5, pad=2),
	        conv3 = L.Convolution2D(256, 384, 3, pad=1),
	        conv4 = L.Convolution2D(384, 384, 3, pad=1),
	        conv5 = L.Convolution2D(384, 256, 3, pad=1),
	        fc6 = L.Linear(None, 4096),
	        fc7 = L.Linear(4096, 4096),
	        fc8 = L.Linear(4096, n_label),
	    )

	def __call__(self, x):
		h = F.max_pooling_2d(F.local_response_normalization(
		    F.relu(self.conv1(x))), 3, stride=2)
		h = F.max_pooling_2d(F.local_response_normalization(
		    F.relu(self.conv2(h))), 3, stride=2)
		h = F.relu(self.conv3(h))
		h = F.relu(self.conv4(h))
		h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
		h = F.dropout(F.relu(self.fc6(h)))
		h = F.dropout(F.relu(self.fc7(h)))
		return self.fc8(h)

class DeepLearningClassifier:
	def __init__(self):
		model = Alex()
		self.model = L.Classifier(model)
		self.opt = optimizers.Adam()
		self.opt.setup(self.model)

	def fit(self,X_train, y_train):
		train_data = tuple_dataset.TupleDataset(X_train, y_train)
		train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
		updater = chainer.training.StandardUpdater(train_iter, self.opt)
		self.trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')
		self.trainer.extend(extensions.LogReport())
		self.trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
		self.trainer.extend(extensions.ProgressBar())
		self.trainer.run()

	def fit_and_score(self, X_train, y_train, X_test, y_test):
		train_data = tuple_dataset.TupleDataset(X_train, y_train)
		test_data = tuple_dataset.TupleDataset(X_test, y_test)
		train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
		test_iter = chainer.iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)
		updater=chainer.training.StandardUpdater(train_iter, self.opt)
		self.trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')
		self.trainer.extend(extensions.Evaluator(test_iter, self.model))
		self.trainer.extend(extensions.LogReport())
		self.trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
		self.trainer.extend(extensions.ProgressBar())
		self.trainer.run()

	def predict(self, X_test):
		x=Variable(X_test)
		y=self.model.predictor(x)
		answer=y.data
		answer=np.argmax(answer, axis=1)
		return answer

	def score(self, X_test, y_test):
		y=self.predict(X_test)
		N=y_test.size
		return 1.0-np.count_nonzero(y-y_test)/N

	def predict_proba(self, X_test):
		x=Variable(X_test)
		y=self.model.predictor(x)
		y=np.exp(y.data)
		H=y.sum(1).reshape(-1,1)
		return np.exp(y)/H

if __name__=='__main__':
	# mnist 使用例
	# 前処理
	mnist = fetch_mldata('MNIST original', data_home=".")
	X = mnist.data
	y = mnist.target
	X =X/X.max()
	X = X.astype(np.float32)
	y = y.astype(np.int32)
	X =X .reshape(70000,1,28,28)  # 必ず（データの総数, channel数, 縦, 横）の形にしておく
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# 定義
	clf = DeepLearningClassifier()
	# トレーニング
	clf.fit(X_train, y_train)
	# 予測
	prediction = clf.predict(X_test)
	# 精度測定
	acc = clf.score(X_test, y_test)

	"""
	トレーニングと予測精度の測定を一度にやってしまいたい場合は
	clf.fit_and_score(X_train, y_train, X_test, y_test)
	各ラベルの確率を計算したいなら
	clf.predict_proba(X_test)
	"""