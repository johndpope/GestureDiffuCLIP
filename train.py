import torch

import torch.nn.functional as F
from torch.utils.data import  DataLoader
from accelerate import Accelerator
import subprocess
from GestureDataset  import GestureDataset,segment_into_sentences,extract_audio_segment,extract_gesture_segment
from model import DenoisingNetwork,StyleEncoder,GestureTranscriptEncoder,VQVAE
import logging
from omegaconf import OmegaConf
from typing import Dict, Any
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Error handling decorator
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


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

def train_gesture_transcript_encoder(encoder, vqvae, dataloader, optimizer, accelerator, config: OmegaConf):
    wandb.init(project=config.wandb.project_name, entity=config.wandb.entity, config=OmegaConf.to_container(config, resolve=True))
    
    for epoch in range(config.training.num_epochs):
        total_loss = 0
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
                gesture_segment = extract_gesture_segment(gesture, config.data.gesture_fps, sentence['start_time'], sentence['end_time'])

                z_q, _, _ = vqvae.encoder(gesture_segment.unsqueeze(0))
                z_g, z_t = encoder(z_q, [' '.join([word for _, _, word in sentence['words']])])
                z_g_list.append(z_g)
                z_t_list.append(z_t)

            z_g = torch.cat(z_g_list, dim=0)
            z_t = torch.cat(z_t_list, dim=0)

            loss = contrastive_loss(z_g, z_t)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch}: Average Loss {avg_loss:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save model checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(encoder)
            accelerator.save(unwrapped_model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
            wandb.save(f"checkpoint_epoch_{epoch+1}.pth")

    wandb.finish()

def main():
    config = OmegaConf.load('config.yaml')
    accelerator = Accelerator()

    # Run MFA
    run_mfa(config)

    # Initialize models
    vqvae = VQVAE(config)
    gesture_transcript_encoder = GestureTranscriptEncoder(config)
    style_encoder = StyleEncoder(config)
    denoising_network = DenoisingNetwork(config)

    # Create dataset and dataloader
    dataset = GestureDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

    # Prepare for distributed training
    vqvae, gesture_transcript_encoder, style_encoder, denoising_network, dataloader = accelerator.prepare(
        vqvae, gesture_transcript_encoder, style_encoder, denoising_network, dataloader
    )

    # Training
    optimizer = torch.optim.Adam(
        list(vqvae.parameters()) + list(gesture_transcript_encoder.parameters()),
        lr=config.training.learning_rate
    )


    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    train_gesture_transcript_encoder(gesture_transcript_encoder, vqvae, dataloader, optimizer, accelerator, config)

if __name__ == "__main__":
    main()