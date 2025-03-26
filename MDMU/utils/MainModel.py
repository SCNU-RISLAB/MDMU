import numpy as np
import torch
from torch import nn
from transformers import RobertaModel
from . import Unetmamba2816
from . import MMF
from mamba_ssm import Mamba
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))  # 可以使其为可学习的参数

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.K = 2048  # queue length
        self.m = 0.999 # momentum
        self.T = 0.07 # tempreture
        self.dim = 128
        self.T_dim = 1024
        self.A_dim = 1024
        self.V_dim = 768
        self.swish = Swish()
        self.MambaLayer_A = nn.Sequential(*[Mamba(d_model=1024, d_state=64, d_conv=4, expand=2) for _ in range(3)]).to("cuda")
        self.MambaLayer_V = nn.Sequential(*[Mamba(d_model=768, d_state=64, d_conv=4, expand=2) for _ in range(3)]).to("cuda")
        self.MambaLayer_A_m = nn.Sequential(*[Mamba(d_model=1024, d_state=64, d_conv=4, expand=2) for _ in range(3)]).to("cuda")
        self.MambaLayer_V_m = nn.Sequential(*[Mamba(d_model=768, d_state=64, d_conv=4, expand=2) for _ in range(3)]).to("cuda")
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.roberta_model_m = RobertaModel.from_pretrained('roberta-large')

        config_U = Unetmamba2816.Configs()
        self.U = Unetmamba2816.Model(config_U)
        self.mmf = MMF.MMF(3, 96, [self.T_dim, self.A_dim, self.V_dim])

        # Lock the parameters of roberta_m
        for param_T, param_T_m in zip(
                self.roberta_model.parameters(), self.roberta_model_m.parameters()
        ):
            param_T_m.data.copy_(param_T.data)  # initialize
            param_T_m.requires_grad = False  # not update by gradient

        # Lock the parameters of mamba_a_m
        for param_A, param_A_m in zip(
                self.MambaLayer_A.parameters(), self.MambaLayer_A_m.parameters()
        ):
            param_A_m.data.copy_(param_A.data)  # initialize
            param_A_m.requires_grad = False  # not update by gradient

        # Lock the parameters of mamba_v_m
        for param_V, param_V_m in zip(
                self.MambaLayer_V.parameters(), self.MambaLayer_V_m.parameters()
        ):
            param_V_m.data.copy_(param_V.data)  # initialize
            param_V_m.requires_grad = False  # not update by gradient

        # Create a queue to store features A and V (keys)
        self.register_buffer("queue", torch.randn(self.dim, self.K*2))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.proj_A = nn.Linear(self.A_dim,128)
        self.proj_A_m = nn.Linear(self.A_dim,128)
        self.proj_V = nn.Linear(self.V_dim,128)
        self.proj_V_m = nn.Linear(self.V_dim,128)
        self.proj_T = nn.Linear(self.T_dim,128)
        self.proj_T_m = nn.Linear(self.T_dim,128)

        self.fused_output_layers = nn.Sequential(
            self.U,
            nn.Dropout(config.dropout),
            nn.Linear(self.T_dim+self.A_dim+self.V_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        # self.fused_output_layers = nn.Sequential(
        #     self.U,
        #     nn.Dropout(config.dropout),
        #     nn.Linear(1024 * 2 + 768 * 2 + 512, 1)
        # )

        self.make_output = nn.Sequential(nn.AvgPool1d(kernel_size=96),
                                         nn.Flatten())

        self.video_pad = nn.Upsample(size=96, mode='linear', align_corners=False)  # Extend the length of the video sequence to 96 (if it were already 96, there would be no change)
        self.audio_pad = nn.Upsample(size=96, mode='linear', align_corners=False)  # Expand the audio sequence length to 96

        self.model_pairs = [[self.roberta_model, self.roberta_model_m],
                            [self.MambaLayer_A, self.MambaLayer_A_m],
                            [self.MambaLayer_V, self.MambaLayer_V_m],
                            [self.proj_T, self.proj_T_m],
                            [self.proj_A, self.proj_A_m],
                            [self.proj_V, self.proj_V_m]]

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.m + param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, audio_feat, video_feat):

        batch_size = audio_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # Ensure that the queue size can be divided by batch size to simplify queue management logic

        # replace the keys at ptr (dequeue and enqueue)
        # Even index stores audio features, odd index stores video features
        self.queue[:, 2 * ptr:2 * ptr + 2 * batch_size:2] = audio_feat.T
        self.queue[:, 2 * ptr + 1:2 * ptr + 2 * batch_size:2] = video_feat.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, video_inputs, video_context_inputs, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                audio_context_inputs):
        v_len = torch.full((16,), 16, dtype=torch.int32)

        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        # input_pooler = raw_output["pooler_output"]
        T_features = raw_output.last_hidden_state

        A_features = self.MambaLayer_A(audio_inputs)
        V_features = self.MambaLayer_V(video_inputs)

        A_features = A_features.permute(0, 2, 1)
        A_features_padded = self.audio_pad(A_features)
        A_features_padded = A_features_padded.permute(0, 2, 1)

        V_features = V_features.permute(0, 2, 1)
        V_features_padded = self.video_pad(V_features)
        V_features_padded = V_features_padded.permute(0, 2, 1)

        fused_features = self.mmf([T_features, A_features_padded, V_features_padded]) # Shape is [batch_size, 96, 5120]
        fused_output = self.fused_output_layers(fused_features)  # Shape is [batch_size, 96, 1]
        fused_output = fused_output.permute(0, 2, 1)
        fused_output = self.make_output(fused_output)

        # get momentum features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update()  # update the key encoder
            T_features_m = self.roberta_model_m(text_inputs, text_mask, return_dict=True).last_hidden_state
            A_features_padded_m = self.MambaLayer_A_m(audio_inputs)
            V_features_padded_m = self.MambaLayer_V_m(video_inputs)

            # Sequence pooling → linear dimensionality reduction → normalization
            T_features_m = F.normalize(self.proj_T(torch.mean(T_features_m, dim=1)), dim=-1)
            A_features_padded_m = F.normalize(self.proj_A_m(torch.mean(A_features_padded_m, dim=1)), dim=-1)
            V_features_padded_m = F.normalize(self.proj_V_m(torch.mean(V_features_padded_m, dim=1)), dim=-1)

        # Sequence pooling → linear dimensionality reduction → normalization
        T_features = F.normalize(self.proj_T(torch.mean(T_features, dim=1)), dim=-1)
        A_features_padded = F.normalize(self.proj_A(torch.mean(A_features_padded, dim=1)), dim=-1)
        V_features_padded = F.normalize(self.proj_V(torch.mean(V_features_padded, dim=1)), dim=-1)


        query = torch.cat([T_features, A_features_padded, V_features_padded], dim=1)
        positive1 = torch.cat([T_features_m, A_features_padded_m,V_features_padded_m], dim=1)
        # positive2 = torch.cat([T_features_m, T_features_m, V_features_padded_m], dim=1)

        # Extract audio and video features from the queue
        audio_queue = self.queue[:, 0::2].clone().detach().T  #  [queue size, dim]
        video_queue = self.queue[:, 1::2].clone().detach().T  #  [queue size, dim]

        # Expand the current batch's T_i and V_i
        T_i_expanded = T_features.unsqueeze(1).expand(-1,self.K , -1)  #  [batch size, queue size, dim]
        V_i_expanded = V_features_padded.unsqueeze(1).expand(-1, self.K, -1)  #  [batch size, queue size, dim]
        A_i_expanded = A_features_padded.unsqueeze(1).expand(-1, self.K, -1)  #  [batch size, queue size, dim]

        # Expand the A_k and V_k in the queue; The subscript k represents the features in the queue
        A_k = audio_queue.unsqueeze(0).expand(self.batch_size, -1, -1)  #  [batch size, queue size, dim]
        V_k = video_queue.unsqueeze(0).expand(self.batch_size, -1, -1)  #  [batch size, queue size, dim]

        # Generate negative sample type one:[T_i, A_k, V_i]
        negative_samples_type1 = torch.cat([T_i_expanded, A_k, V_i_expanded], dim=2)  # [batch size, queue size, 3*dim]

        # Generate negative sample type two:[T_i, A_i, V_k]
        negative_samples_type2 = torch.cat([T_i_expanded, A_i_expanded, V_k], dim=2)  # # [batch size, queue size, 3*dim]

        # Merge all negative samples, [batch_size, queue size, 3*dim]
        negative_keys = torch.cat([negative_samples_type1, negative_samples_type2], dim=1)  # # [batch size, queue size, 3*dim]
        negative_keys = nn.functional.normalize(negative_keys, dim=2)  # [batch size, queue size, 3*dim]

        # Calculate the similarity between the query and the positive sample, [batch_size, 1]
        logits_pos1 = torch.einsum("nc,nc->n", [ query, positive1]).unsqueeze(-1)
        # logits_pos2 = torch.einsum("nc,nc->n", [ query, positive2]).unsqueeze(-1)

        # Calculate the similarity between the query and the negtive sample,  [batch_size, K]
        logits_neg = torch.bmm(negative_keys, query.unsqueeze(2)).squeeze(2)  #  [batch_size, K]

        # concat logits, [batch_size, 1 + K]
        logits1 = torch.cat([logits_pos1, logits_neg], dim=1)  #  [batch_size, 1 + K]
        # logits2 = torch.cat([logits_pos2, logits_neg], dim=1)  #  [batch_size, 1 + K]

        logits1 /= self.T
        # logits2 /= self.T

        # Set label, shape [N,]
        labels = torch.zeros(self.batch_size, dtype=torch.long).cuda()

        loss1 = nn.CrossEntropyLoss()(logits1, labels)
        # loss2 = nn.CrossEntropyLoss()(logits2, labels)

        # loss = (loss1 + loss2) / 2
        loss = loss1

        # update queue
        self._dequeue_and_enqueue(A_features_padded_m, V_features_padded_m)

        return {
            # 'T': T_output,
            # 'A': A_output,
            'M': fused_output,
            'CL_loss': loss,
            'embedding':fused_features.cpu()
        }

