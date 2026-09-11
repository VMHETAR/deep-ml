def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'row':
		means = []

		for i in range(len(matrix)):
			means.append((sum(matrix[i]))/len(matrix[i]))
	elif mode == 'column':
		means = []
		for i in range(len(matrix[0])):
			total = 0
			for j in range(len(matrix)):
				total += matrix[j][i]
			means.append((total/len(matrix)))
	return means