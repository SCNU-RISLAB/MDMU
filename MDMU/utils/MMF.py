import torch
import torch.nn as nn
import torch.nn.functional as F
import shap


class MMF(nn.Module):
    def __init__(self, num_modalities, sequence_length, channel_sizes):
        super(MMF, self).__init__()
        self.num_modalities = num_modalities
        self.sequence_length = sequence_length
        self.channel_sizes = channel_sizes  # List of channel numbers for each modality
        # Create a linear layer for each modality to generate scalar weights
        self.weight_generators = nn.ModuleList(
            [nn.Linear(channel_size, 1) for channel_size in channel_sizes]
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x should be a list containing sequence data for each modality
        if not isinstance(x, list) or len(x) != self.num_modalities:
            raise ValueError("Input must be a list of tensors with length equal to 'num_modalities'.")

        modal_weights = []
        for generator, modality_data in zip(self.weight_generators, x):
            # modality_data: (batch_size, sequence_length, channel_size_i)
            batch_size, sequence_length, channel_size_i = modality_data.shape
            # Reshape the data into two dimensions for linear layer processing
            modality_data_reshaped = modality_data.reshape(-1, channel_size_i)
            w = generator(modality_data_reshaped)
            w = self.sigmoid(w)  # (batch_size * sequence_length, 1)
            scalar_weight = w.mean()
            modal_weights.append(scalar_weight)

        # Stack scalar weights and normalize them using softmax
        scalar_weights = torch.stack(modal_weights)  # (num_modalities,)
        tempreture = 1
        normalized_weights = F.softmax(scalar_weights/tempreture, dim=0)  # (num_modalities,)

        print("Global Weights for each modality:", normalized_weights)

        # Weight the data of each modality according to normalized scalar weights
        weighted_modalities = [x[i] * normalized_weights[i] for i in range(self.num_modalities)]
        # Splicing weighted modal data in the channel dimension
        fused_output = torch.cat(weighted_modalities, dim=2)

        return fused_output

class MMF2(nn.Module):
    def __init__(self, num_modalities, sequence_length, channel_sizes):
        super(MMF2, self).__init__()
        self.num_modalities = num_modalities
        self.sequence_length = sequence_length
        self.channel_sizes = channel_sizes

    def forward(self, x):
        if not isinstance(x, list) or len(x) != self.num_modalities:
            raise ValueError("Input must be a list of tensors with length equal to 'num_modalities'.")

        modal_outputs = x

        scalar_weights = []
        for modality_data in modal_outputs:
            explainer = shap.Explainer(modality_data)
            shap_values = explainer(modality_data)
            scalar_weight = shap_values.mean()
            scalar_weights.append(scalar_weight)

        scalar_weights = torch.tensor(scalar_weights)
        normalized_weights = torch.softmax(scalar_weights, dim=0)

        weighted_modalities = [modal_outputs[i] * normalized_weights[i] for i in range(self.num_modalities)]
        fused_output = torch.cat(weighted_modalities, dim=2)

        return fused_output

if __name__=='__main__':

    num_modalities = 3
    sequence_length = 96
    channel_sizes = [50, 30, 80]
    mmf = MMF(num_modalities, sequence_length, channel_sizes)

    features = [torch.randn(1, sequence_length, channel_sizes[i]) for i in range(num_modalities)]

    # 传递序列数据到MSF层
    output = mmf(features)
    print("Output shape:", output.shape)

