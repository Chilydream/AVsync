代码位置：248:/root/ChineseDataset/AVsync

yolo模型调用说明：
```python
import sys
sys.path.append('./third_party/yolo')
import torch
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect

model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt'))

batch_size = 8
img_batch = torch.randn((batch_size, 3, 256, 256))
bbox_list = face_detect(model_yolo, img_batch)
```

hrnet模型调用说明：
```python
import sys
sys.path.append('./third_party/HRNet')
import torch
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
model_hrnet = model_hrnet.eval()

batch_size = 8
face_batch = torch.randn((batch_size, 3, 256, 256))
lmk_list = get_batch_lmks(model_hrnet, face_batch)
```

匹配模型调用说明：
```python
import torch
from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGG6_speech
from model.SyncModel import SyncModel


model_lmk2lip = Lmk2LipModel(lmk_emb=40, lip_emb=256, stride=1)
model_wav2v = VGG6_speech(n_out=256)
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