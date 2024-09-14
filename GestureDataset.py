import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import librosa
from praatio import tgio
import torchaudio
import os
import subprocess


class GestureDataset(Dataset):
    def __init__(self, data_dir, audio_dir, transcript_dir, alignment_dir, gesture_dir):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.alignment_dir = alignment_dir
        self.gesture_dir = gesture_dir
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for file in os.listdir(self.alignment_dir):
            if file.endswith('.TextGrid'):
                base_name = os.path.splitext(file)[0]
                samples.append({
                    'audio': os.path.join(self.audio_dir, f"{base_name}.wav"),
                    'transcript': os.path.join(self.transcript_dir, f"{base_name}.txt"),
                    'alignment': os.path.join(self.alignment_dir, file),
                    'gesture': os.path.join(self.gesture_dir, f"{base_name}.npy")
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = torchaudio.load(sample['audio'])
        with open(sample['transcript'], 'r') as f:
            transcript = f.read().strip()
        gesture = np.load(sample['gesture'])
        alignment = load_alignment(sample['alignment'])
        return {
            'audio': audio,
            'sr': sr,
            'transcript': transcript,
            'gesture': torch.tensor(gesture, dtype=torch.float32),
            'alignment': alignment
        }
def load_alignment(textgrid_path):
    tg = tgio.openTextgrid(textgrid_path)
    word_tier = tg.tierDict['words']
    return [(entry.start, entry.end, entry.label) for entry in word_tier.entryList]

def segment_into_sentences(word_entries):
    sentences = []
    current_sentence = []
    sentence_start = None

    for idx, (start_time, end_time, word) in enumerate(word_entries):
        if sentence_start is None:
            sentence_start = start_time
        current_sentence.append((start_time, end_time, word))
        if word in ['.', '!', '?'] or idx == len(word_entries) - 1:
            sentence_end = end_time
            sentences.append({
                'words': current_sentence,
                'start_time': sentence_start,
                'end_time': sentence_end
            })
            current_sentence = []
            sentence_start = None

    return sentences

def extract_audio_segment(audio_waveform, sample_rate, start_time, end_time):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return audio_waveform[:, start_sample:end_sample]

def extract_gesture_segment(gesture_data, frame_rate, start_time, end_time):
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    return gesture_data[start_frame:end_frame]