TRAIN_PARAMETER = {
	'mode': 'train',
	'pretrain_model': None,
	'exp_dir': 'exp',
	'exp_num': '0',
	'save_step': 1,
	'valid_step': 1000,

	'epoch': 1000,
	'batch_size': 4,
	'num_workers': 16,
	'resolution': 256,
	'lr': 1e-2,
	'lr_gamma': 0.9,
	'gpu': True,
	'wandb': False,

	'train_list': "metadata/train.txt",
	'val_list': 'metadata/val.txt',
	'test_list': 'metadata/test.txt',
	'face_eb': 256,
	'voice_eb': 256,
	'n_mfcc': 39,
}
