import numpy as np

card_list = ['00:0', '00:1']
for i in range(1, 14):
    for j in range(4):
        card_list.append(f'{i:02d}:{j+1}')

hand_list = np.random.choice(card_list, 20, replace=False)
hand_list.sort()
print(hand_list)