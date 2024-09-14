
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = VQVAEEncoder(input_dim, hidden_dim, num_embeddings, embedding_dim)
        self.decoder = VQVAEDecoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, x):
        z_q, encoding_indices, z_e = self.encoder(x)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, encoding_indices, z_e

class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(VQVAEEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        z_e = F.relu(self.conv1(x))
        z_e = self.conv2(z_e)
        z_e_flattened = z_e.permute(0, 2, 1).contiguous().view(-1, z_e.size(1))
        distances = torch.cdist(z_e_flattened, self.codebook.weight, p=2)
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(encoding_indices).view(z_e.shape)
        return z_q, encoding_indices, z_e

class VQVAEDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(VQVAEDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, z_q):
        x = F.relu(self.conv1(z_q))
        x_recon = self.conv2(x)
        return x_recon

class GestureTranscriptEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GestureTranscriptEncoder, self).__init__()
        self.gesture_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=6
        )
        self.transcript_encoder = T5EncoderModel.from_pretrained('t5-base')
        self.transcript_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def forward(self, z_q, transcripts):
        z_g = self.gesture_encoder(z_q.permute(2, 0, 1)).permute(1, 2, 0)
        z_g_pooled = torch.max(z_g, dim=2)[0]

        inputs = self.transcript_tokenizer(transcripts, return_tensors='pt', padding=True, truncation=True)
        z_t = self.transcript_encoder(input_ids=inputs.input_ids).last_hidden_state
        z_t_pooled = torch.max(z_t, dim=1)[0]

        return z_g_pooled, z_t_pooled

class StyleEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(StyleEncoder, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.text_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_encoder = CLIPModel.from_pretrained(model_name).vision_model
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, style_prompt, prompt_type):
        if prompt_type == 'text':
            inputs = self.text_tokenizer(style_prompt, return_tensors='pt', padding=True, truncation=True)
            outputs = self.text_encoder(input_ids=inputs.input_ids)
            return outputs.last_hidden_state.mean(dim=1)
        elif prompt_type == 'image':
            inputs = self.processor(images=style_prompt, return_tensors='pt')
            outputs = self.image_encoder(pixel_values=inputs.pixel_values)
            return outputs.pooler_output
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

class DenoisingNetwork(nn.Module):
    def __init__(self, d_model, nhead, num_layers, style_dim):
        super(DenoisingNetwork, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4),
            num_layers
        )
        self.adain = AdaIN(d_model, style_dim)

    def forward(self, z_noisy, audio, transcript, style):
        z = self.transformer(z_noisy.permute(2, 0, 1))
        z = self.adain(z.permute(1, 2, 0), style)
        return z

class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
    


