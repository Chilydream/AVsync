import torch
import ipdb


class VoiceModel(torch.nn.Module):
    def __init__(self, voice_encoder, emb_dim, people_count):
        super(VoiceModel, self).__init__()
        self.voice_encoder = voice_encoder
        self.matrix = torch.nn.Linear(in_features=emb_dim, out_features=people_count, bias=True)

    def forward(self, voice):
        emb_voice = self.voice_encoder(voice)
        logits = self.matrix(emb_voice)
        return logits


class FaceModel(torch.nn.Module):
    def __init__(self, face_encoder, emb_dim, people_count):
        super(FaceModel, self).__init__()
        self.voice_encoder = face_encoder
        self.matrix = torch.nn.Linear(in_features=emb_dim, out_features=people_count, bias=True)

    def forward(self, voice):
        emb_voice = self.voice_encoder(voice)
        logits = self.matrix(emb_voice)
        return logits
