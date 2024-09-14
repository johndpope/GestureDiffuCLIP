import torch

import torch.nn.functional as F
from torch.utils.data import  DataLoader
from accelerate import Accelerator
import subprocess
from GestureDataset  import GestureDataset,segment_into_sentences,extract_audio_segment,extract_gesture_segment
from model import DenoisingNetwork,StyleEncoder,GestureTranscriptEncoder,VQVAE

def train_vqvae(vqvae, dataloader, optimizer, accelerator, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch['motion']
            x_recon, z_q, _, z_e = vqvae(x)
            loss = F.mse_loss(x_recon, x) + F.mse_loss(z_q.detach(), z_e) + F.mse_loss(z_q, z_e.detach())
            accelerator.backward(loss)
            optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")


def train_denoising_network(denoising_network, vqvae, style_encoder, dataloader, optimizer, accelerator, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch['motion']
            audio = batch['audio']
            transcript = batch['transcript']
            style_prompt = batch['style_prompt']

            z_q, _, _ = vqvae.encoder(x)
            noise = torch.randn_like(z_q)
            t = torch.randint(0, 1000, (z_q.size(0),))
            z_noisy = z_q + noise * t.view(-1, 1, 1)

            style = style_encoder(style_prompt, prompt_type='text')
            z_denoised = denoising_network(z_noisy, audio, transcript, style)

            loss = F.mse_loss(z_denoised, z_q)
            accelerator.backward(loss)
            optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

def contrastive_loss(z_g, z_t, temperature=0.07):
    z_g = F.normalize(z_g, dim=1)
    z_t = F.normalize(z_t, dim=1)
    logits = torch.mm(z_g, z_t.t()) / temperature
    labels = torch.arange(z_g.size(0), device=z_g.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss / 2



def run_mfa(audio_dir, transcript_dir, output_dir, model='english'):
    cmd = f"mfa align {audio_dir} {transcript_dir} {model} {output_dir}"
    subprocess.run(cmd, shell=True, check=True)


def train_gesture_transcript_encoder(encoder, vqvae, dataloader, optimizer, accelerator, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            audio = batch['audio']
            transcript = batch['transcript']
            gesture = batch['gesture']
            alignment = batch['alignment']

            sentences = segment_into_sentences(alignment)
            z_g_list, z_t_list = [], []

            for sentence in sentences:
                audio_segment = extract_audio_segment(audio, batch['sr'], sentence['start_time'], sentence['end_time'])
                gesture_segment = extract_gesture_segment(gesture, 60, sentence['start_time'], sentence['end_time'])  # Assuming 60 FPS

                z_q, _, _ = vqvae.encoder(gesture_segment.unsqueeze(0))
                z_g, z_t = encoder(z_q, [' '.join([word for _, _, word in sentence['words']])])
                z_g_list.append(z_g)
                z_t_list.append(z_t)

            z_g = torch.cat(z_g_list, dim=0)
            z_t = torch.cat(z_t_list, dim=0)

            loss = contrastive_loss(z_g, z_t)
            accelerator.backward(loss)
            optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item()}")

def main():
    accelerator = Accelerator()

    # Run MFA (assuming it's installed and audio/transcript files are prepared)
    run_mfa('path/to/audio', 'path/to/transcripts', 'path/to/alignments')

    # Initialize models
    vqvae = VQVAE(input_dim=135, hidden_dim=256, num_embeddings=1024, embedding_dim=512)
    gesture_transcript_encoder = GestureTranscriptEncoder(embedding_dim=512, hidden_dim=1024)
    style_encoder = StyleEncoder()
    denoising_network = DenoisingNetwork(d_model=512, nhead=8, num_layers=6, style_dim=512)

    # Create dataset and dataloader
    dataset = GestureDataset('path/to/data', 'path/to/audio', 'path/to/transcripts', 'path/to/alignments', 'path/to/gestures')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Prepare for distributed training
    vqvae, gesture_transcript_encoder, style_encoder, denoising_network, dataloader = accelerator.prepare(
        vqvae, gesture_transcript_encoder, style_encoder, denoising_network, dataloader
    )

    # Training
    optimizer = torch.optim.Adam(list(vqvae.parameters()) + list(gesture_transcript_encoder.parameters()), lr=1e-4)

    train_gesture_transcript_encoder(gesture_transcript_encoder, vqvae, dataloader, optimizer, accelerator, num_epochs=50)

if __name__ == "__main__":
    main()
