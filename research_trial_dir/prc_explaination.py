

precision, recall, prc_threshold = [0.1, 0.2, 0.4, 0.9, 0.99, 0.22, 0.66], [0.1, 0.2, 0.4, 0.9, 0.99, 0.22, 0.66], [0.1, 0.2, 0.4, 0.9, 0.99, 0.22, 0.66]

n_th = 4

print(list(zip(precision, recall, prc_threshold))[::n_th])