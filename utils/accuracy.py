import torch
import numpy as np


def topk_acc(score_mat, label, k=1):
	# 要求传进来的label是tensor
	max_idx = score_mat.topk(k, dim=1)
	correct_cnt = 0
	total_cnt = len(max_idx[1])
	for i in range(total_cnt):
		# max_idx[1][i][1]
		# 第一个 1 指的是 topk的序号（0对应的是值）
		# 第二个 i 指的是 第i个样本
		# max_idx[1][i]是一个list，表示topk的元素对应的标签值序列
		if label[i] in label[max_idx[1][i]]:
			correct_cnt += 1
	return correct_cnt/total_cnt


def get_new_idx(batch_size):
	new_idx = np.arange(0, batch_size)
	np.random.shuffle(new_idx)
	# new_idx[:batch_size//2] = range(batch_size//2)
	# 半个 batch的是匹配的音脸，另外半个随机
	return new_idx


def get_rand_idx(batch_size):
	a = np.random.randint(0, 2)
	new_idx = np.arange(0, batch_size)
	if a == 0:
		new_idx[:batch_size//2] = new_idx[batch_size//2:]
	else:
		new_idx[batch_size//2:] = new_idx[:batch_size//2]
	return new_idx


def get_gt_label(wid_label: torch.Tensor, new_idx):
	assert len(wid_label) == len(new_idx), ValueError(f'Size of wid_label should match new_idx, '
	                                                  f'but got {len(wid_label)} and {len(new_idx)}')
	gt_label = torch.where(wid_label == wid_label[new_idx], 1., -1.)
	return gt_label
