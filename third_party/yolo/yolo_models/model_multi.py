import torch
import ipdb


class MultiFrame(torch.nn.Module):
    def __init__(self, voice_encoder, face_encoder, fuser, freeze_face, freeze_voice):
        super(MultiFrame, self).__init__()
        self.voice_encoder = voice_encoder
        self.face_encoder = face_encoder
        self.freeze_face = freeze_face
        self.freeze_voice = freeze_voice
        self.fuser = fuser

    def forward(self, voice, face_list):
        # voice:
        if self.freeze_voice:
            with torch.no_grad():
                emb_voice = self.voice_encoder(voice)
        else:
            emb_voice = self.voice_encoder(voice)

        # face：
        if self.freeze_face:
            with torch.no_grad():
                batch_face_emb = self.encode_face(face_list)
        else:
            batch_face_emb = self.encode_face(face_list)

        face_emb = self.fuser(batch_face_emb)
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


class GRUFuser(torch.nn.Module):
    def __init__(self):
        super(GRUFuser, self).__init__()
        self.gru = torch.nn.GRU(input_size=512,
                                hidden_size=256,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, batch_face_emb):
        _, h_n = self.gru(batch_face_emb)
        face_emb = self.get_last_emb(h_n, 1, 2)
        return face_emb

    def get_last_emb(self, hidden_states_3dim, num_rnn_layer, num_direction=2):
        # (num_directions x num_layers， batch_size，hidden_size)

        _, batch_size, rnn_hidden_size = hidden_states_3dim.shape

        hidden_states_4dim = hidden_states_3dim.view(num_rnn_layer, num_direction, batch_size, rnn_hidden_size)
        # (num_layers,num_directions,batch_size,hidden_size)

        last_layer = hidden_states_4dim[num_rnn_layer - 1]
        last_layer_concat = torch.cat([last_layer[0], last_layer[1]], dim=1)
        embedding = last_layer_concat
        return embedding


class AvgPoolFuser(torch.nn.Module):
    def __init__(self):
        super(AvgPoolFuser, self).__init__()

    def forward(self, batch_face_emb):
        face_emb = torch.mean(batch_face_emb, dim=1)
        return face_emb


class AttentionFuser(torch.nn.Module):
    def __init__(self):
        super(AttentionFuser, self).__init__()

        self.matrix1 = torch.nn.Linear(512, 30)

    def forward(self, batch_face_emb):
        # (batch,seq_length,feature_size)

        return face_emb


if __name__ == "__main__":
    batch_face_emb = torch.rand((4,10,128))

    matrix1 = torch.nn.Linear(128,128)

    pass

