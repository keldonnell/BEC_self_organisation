import numpy as np

digits = []
digit_length = 20 

for i in range(digit_length):
    digits.append(np.random.randint(0, 10))

print(digits)

len_adj = 13
current_mult = 1 
max_mult = 0  

# def mult_next_adj_nonzero_set(start_i, end_i):
#     mult = 1
#     for i in range(start_i, end_i):
#         if digits[i] == 0:
#             mult, end_i = mult_next_adj_nonzero_set(i + 1, end_i + 1)
#             break
#         mult *= digits[i]
#     return mult, end_i

# for start_i in range(len(digits)):
#     mult, end_i = mult_next_adj_nonzero_set(start_i, start_i + len_adj)
#     if mult > max_mult:
#         max_mult = mult
#     if end_i >= start_i + len_adj:
#         break
#     start_i = end_i - 1

# print(max_mult)


