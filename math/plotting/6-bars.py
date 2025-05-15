#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))


people = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4'] 

fig, ax = plt.subplots(figsize=(8, 6))


x = np.arange(len(people))


bottom = np.zeros(len(people))

for i, (fruit_name, color) in enumerate(zip(fruits, colors)):
    ax.bar(x, fruit[i], bottom=bottom, width=0.5, label=fruit_name, color=color)
    bottom += fruit[i]


ax.set_title('Number of Fruit per Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_xticks(x)
ax.set_xticklabels(people)
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))

ax.legend()

plt.tight_layout()
plt.show()