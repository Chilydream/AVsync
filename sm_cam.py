from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

from model.MultiSensory import MultiSensory

model_ms = MultiSensory(sound_rate=16000, image_fps=25)
target_layer = model_ms.img_block0
