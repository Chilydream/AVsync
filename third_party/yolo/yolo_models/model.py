import torch
import ipdb


class Model(torch.nn.Module):
    def __init__(self, voice_encoder, face_encoder,lock_voice=False):
        super(Model, self).__init__()
        self.voice_encoder = voice_encoder
        self.face_encoder = face_encoder
        self.lock_voice = lock_voice

    def forward(self, voice, face1):
        if self.lock_voice:
            with torch.no_grad():
                emb_voice = self.voice_encoder(voice)
        else:
            emb_voice = self.voice_encoder(voice)
        emb_face1 = self.face_encoder(face1)
        return emb_voice, emb_face1


class MultiFrame(torch.nn.Module):
    def __init__(self, voice_encoder, face_encoder, freeze_face=False):
        super(MultiFrame, self).__init__()
        self.voice_encoder = voice_encoder
        self.face_encoder = face_encoder
        self.freeze_face = freeze_face
        self.gru = torch.nn.GRU(input_size=512,
                                hidden_size=256,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, voice, face_list):
        emb_voice = self.voice_encoder(voice)

        # 脸：
        if self.freeze_face:
            with torch.no_grad():
                batch_face_emb = self.encode_face(face_list)
        else:
            batch_face_emb = self.encode_face(face_list)

        _, h_n = self.gru(batch_face_emb)
        face_emb = get_last_emb(h_n, 1, 2)
        return emb_voice, face_emb

    def encode_face(self, face_list):
        # face_list
        # [tensor(batch_size,....),tensor(),tensor() ]

        batch_size = face_list[0].shape[0]

        if batch_size == 1:
            # 单帧的情况,
            tmp = torch.cat(face_list)
            # [32, 1, 3, 112, 112]

            batch_face_emb = self.face_encoder(tmp)
            # [32, 512]

            batch_face_emb = batch_face_emb.unsqueeze(dim=1)
            # [32, 1, 512]

        else:
            face_emb_list = [self.face_encoder(face) for face in face_list]
            # list长度32,单个元素的尺寸:(frame_size,512)
            batch_face_emb = torch.stack(face_emb_list)
        return batch_face_emb


def get_last_emb(hidden_states_3dim, num_rnn_layer, num_direction=2):
    # (num_directions x num_layers， batch_size，hidden_size)

    _, batch_size, rnn_hidden_size = hidden_states_3dim.shape

    hidden_states_4dim = hidden_states_3dim.view(num_rnn_layer, num_direction, batch_size, rnn_hidden_size)
    # (num_layers,num_directions,batch_size,hidden_size)

    last_layer = hidden_states_4dim[num_rnn_layer - 1]
    last_layer_concat = torch.cat([last_layer[0], last_layer[1]], dim=1)
    embedding = last_layer_concat
    return embedding
