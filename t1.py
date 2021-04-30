# Processamento de Imagens
# Trabalho 1
# Caio Ferreira Bernardo - NUSP 9276936
# João Pedro Hannauer - NUSP 9390486

import numpy as np
import math
import random
import imageio

def normalize(matrix, start, end):
	return (matrix - matrix.min())/(matrix.max() - matrix.min()) * (end-start) + start

def downsampling(matrix, N, C):
	 return matrix[0::math.floor(C/N),0::math.floor(C/N)]

def quantization(matrix, N, B):
	m_normalized = normalize(matrix, 0, 255)
	m_normalized = m_normalized.astype(np.uint8)
	r_shift = np.right_shift(m_normalized, 8-B)
	return np.left_shift(r_shift, 8-B)

def RSE(g, R):
	return np.sqrt(np.sum((g-R)**2))
	# return np.linalg.norm(g-R)

def func1(C):
	return np.fromfunction(lambda x,y: ((x*y)+ 2*y), (C, C))

def func2(C, Q):
	return np.fromfunction(lambda x,y,Q: np.absolute(((np.cos(x/Q))+ (2*np.sin(y/Q)))), (C, C), Q = Q)

def func3(C, Q):
	return np.fromfunction(lambda x,y,Q: np.absolute(((3*(x/Q)) - (np.power((y/Q),(1/3))))), (C, C), Q = Q)

def func4(C, S):
	random.seed(S)
	img = np.zeros((C,C))
	
	for i in range(C):
		for j in range(C):
			img[j,i] = random.random()
	
	return img

def func5(C, S):
	random.seed(S)

	img = np.zeros((C,C))
	
	x = 0
	y = 0
	img[x,y] = 1
	
	for i in range (1 + pow(C,2)):

		x = (x + random.randint(-1, 1)) % C
		y = (y + random.randint(-1, 1)) % C

		img[x,y] = 1

	return img

def main():
	
	# 1. Get user inputs
	filename = input().rstrip()
	C = int(input().rstrip())
	F = int(input().rstrip())
	Q = int(input().rstrip())
	N = int(input().rstrip())
	B = int(input().rstrip())
	S = int(input().rstrip())

	# 2. Select function according to input
	if F == 1:
		m = func1(C)
	elif F == 2:
		m = func2(C, Q)
	elif F == 3:
		m = func3(C, Q)
	elif F == 4:
		m = func4(C, S)
	elif F == 5:
		m = func5(C, S)

	# 3. Sampling and Quantization
	m = normalize(m, 0, 65535)
	m = downsampling(m, N, C)
	result = quantization(m, N, B)

	# 4. Load image and calculate error
	R = np.load(filename)
	error = RSE(result, R)

	print("{:.4f}".format(error))


if __name__ == "__main__":
	main()