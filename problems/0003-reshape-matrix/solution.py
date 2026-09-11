import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	if len(a) is 0:
		return []
	if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
    	return []

	flat = []
	for row in a:
		for x in row:
			flat.append(x)
	reshaped_matrix = []
	for i in range(new_shape[0]):
		row = []
		for j in range(new_shape[1]):
			row.append(flat[i * new_shape[1] + j])
		reshaped_matrix.append(row)
	return reshaped_matrix