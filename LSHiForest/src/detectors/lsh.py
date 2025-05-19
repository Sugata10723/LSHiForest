#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import zlib

class LSH():
	''' The base class of LSH families '''
	def __init__(self, default_pool_size=50):
		self._default_pool_size = default_pool_size

	# Virtual methods
	# type <- get_lsh_type(self) 
	# display_hash_func_parameters(self)
	# x' <- format_for_lsh(self, x)
	# key <- get_hash_value(self, x, hash_index)


class E2LSH(LSH):
	''' Class to build E2 locality sensitive hashing family '''

	def __init__(self, bin_width=4, norm=2, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._dimensions = -1
		self._bin_width = bin_width
		self._norm = norm
		self.A_array = None
		self.B_array = None 


	def get_lsh_type(self):
		return 'L'+str(self._norm)+'LSH'


	def display_hash_func_parameters(self):
		for i in range(len(self.A_array)):
			print (self.A_array[i], self.B_array[i])

	def fit(self, data):
		if (data is None).all():
			return
		self._dimensions = len(data[0])-1

		self.A_array = []
		self.B_array = [] 
		if self._norm == 1:
			self.A_array.append(np.random.standard_cauchy(self._dimensions))
		elif self._norm == 2:
			self.A_array.append(np.random.normal(0.0, 1.0, self._dimensions))
		self.B_array.append(np.random.uniform(0.0, self._bin_width))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated is True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, len(self.A_array)):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated is False:	
					self.A_array.append(a)
					self.B_array.append(b)	


	def format_for_lsh(self, x):
		return x


	def get_hash_value(self, x, hash_index):
		cur_len = len(self.A_array)
		while hash_index >= cur_len:
			repeated = True
			while repeated is True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, cur_len):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated is False:
					self.A_array.append(a)
					self.B_array.append(b)
					cur_len += 1
		return int(np.floor((np.dot(x, self.A_array[hash_index])+self.B_array[hash_index])/self._bin_width))
		

class AngleLSH(LSH):
	def __init__(self, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._weights = None

	def get_lsh_type(self):
		return 'AngleLSH'

	def display_hash_func_parameters(self):
		for i in range(len(self._weights)):
			print(self._weights[i])

	def fit(self, data):
		#if data == None:
		if data is None:
			return
		self._dimensions = len(data[0])-1

		self._weights=[]
		#self._weights.append(np.random.uniform(-1.0, 1.0, self._dimensions))
		self._weights.append(np.random.normal(0.0, 1.0, self._dimensions))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated is True:
				repeated = False
				weight=np.random.uniform(-1.0, 1.0, self._dimensions)
				for j in range(0, len(self._weights)):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated is False:	
					self._weights.append(weight)

	def format_for_lsh(self, x):
		return x

	def get_hash_value(self, x, hash_index):
		cur_len = len(self._weights)
		while hash_index >= cur_len:
			repeated = True
			while repeated is True:
				repeated = False
				#weight=np.random.uniform(-1.0, 1.0, self._dimensions)
				weight=np.random.normal(0.0, 1.0, self._dimensions)
				for j in range(0, cur_len):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated is False:	
					self._weights.append(weight)
					cur_len += 1

		return -1 if np.dot(x, self._weights[hash_index]) <0 else 1


class MinLSH(LSH):
	''' Class to build LSH for set similarity'''	
	def __init__(self, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._mod_base = None
        # zlib.crc32 のシード値として乱数をプール
		self._hash_seeds = np.random.randint(0, 2**32, size=(default_pool_size,), dtype=np.uint32)
	
	def get_lsh_type(self):
		return 'MinLSH'

	def display_hash_func_parameters(self):
		for i in range(len(self._hash_seeds)):
			print(self._hash_seeds[i])
	
	def fit(self, data):
		"""
		Input
        	data: 
        """
        # 全トークン数を数えて prime mod を決定
		unique_tokens = {tok for x in data for tok in x}
		self._mod_base = len(unique_tokens)
		


	def format_for_lsh(self, x):# 入力データはlist?
		'''
		Input
			- x: ndarray or matrix
		Output
			- x: ndarray or matrix 
		'''
		return x

	def get_hash_value(self, x, hash_index): # x:各シグネチャ
		cur_len = len(self._hash_seeds) # 現在のpoolのサイズ数を確認
		while cur_len <= hash_index : # hashが回りすぎてプールされているパラメタが足りない場合
			repeated = True # 生成したパラメタが重複していないかチェックするインディケータ
			while repeated is True:
				repeated = False
				new_para = np.random.randint(0, 2**32, size=(1,), dtype=np.uint32)#　新しいパラメタを生成
				for j in range(0, cur_len):
					if np.array_equal(new_para, self._hash_seeds[j]): # これまでと同じパラメタがあった場合
						repeated = True # 新しいパラメタを廃棄してもういちど生成
						break
				if repeated is False:
					self._hash_seeds = np.concatenate([self._hash_seeds, new_para])
					cur_len += 1

		seed = int(self._hash_seeds[hash_index])
		min_h = np.iinfo('uint32').max
		for idx, token in enumerate(x):
			token = f"{idx}={token}"
			# そのまま文字列をバイト列にして CRC32
			h = zlib.crc32(token.encode('utf-8'), seed) & 0xffffffff
			# prime mod → 次元数以内に落とし込み
			h = h % self._mod_base
			if h < min_h:
				min_h = h
		return int(min_h)

	