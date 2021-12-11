代码位置：248:/root/ChineseDataset/AVsync

模型调用说明：
```python
import torch
from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGGVoice
from model.SyncModel import SyncModel


model_lmk2lip = Lmk2LipModel(lmk_emb=40, lip_emb=256, stride=1)
model_wav2v = VGGVoice(n_out=256)
model_sync = SyncModel(lip_emb=256, voice_emb=256)

model_ckpt = torch.load('pretrain_model/model_LRW.pth')
model_lmk2lip.load_state_dict(model_ckpt['model_lmk2lip'])
model_wav2v.load_state_dict(model_ckpt['model_wav2v'])
model_sync.load_state_dict(model_ckpt['model_sync'])

batch_size = 128
seq_len = 29
lmk_tensor = torch.randn((batch_size, seq_len, 40))
wav_tensor = torch.randn((batch_size, 19456))
lip_emb = model_lmk2lip(lmk_tensor)
voice_emb = model_wav2v(wav_tensor)
# voice_emb = (batch_size, 256)
sync_pred = model_sync(lip_emb, voice_emb)
```