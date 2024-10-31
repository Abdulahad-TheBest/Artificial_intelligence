# 31.10.2024

# Qaror daraxtlari yordamida ma'lumotlar entropiyasini hisoblash

import math
from collections import Counter  # Counter ni import qilish

def entropy(probs):
    return -sum([p * math.log2(p) for p in probs if p > 0])

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

# Example data: [('Green', 'A'), ('Yellow', 'B'), ('Red', 'A'), ('Red', 'B'), ('Yellow', 'A')]
data = [('Green', 'A'), ('Yellow', 'B'), ('Red', 'A'), ('Red', 'B'), ('Yellow', 'A')]
print("Entropy of data:", data_entropy(data))
