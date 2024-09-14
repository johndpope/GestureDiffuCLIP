import argparse
import datetime
import json
from pathlib import Path

from helpers import save_useful_info
from train import train


def train_zeggs():
    # Setting parser
    parser = argparse.ArgumentParser(description="Train ZEGGS Network.")

    # Hparams
    parser.add_argument(
        "-o",
        "--options",
        type=str,
        help="Options filename",
    )
    parser.add_argument('-n', '--name', type=str, help="Name", required=False)

    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = json.load(f)
    if args.name:
        options["name"] = args.name

    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]

    base_path = Path(paths["base_path"])
    path_processed_data = base_path / paths["path_processed_data"] / "processed_data.npz"
    path_data_definition = base_path / paths["path_processed_data"] / "data_definition.json"

    # Output directory
    if paths["output_dir"] is None:
        output_dir = (base_path / "outputs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_dir.mkdir(exist_ok=True, parents=True)
        paths["output_dir"] = str(output_dir)
    else:
        output_dir = Path(paths["output_dir"])

    # Path to models
    if paths["models_dir"] is None and not train_options["resume"]:
        models_dir = output_dir / "saved_models"
        models_dir.mkdir(exist_ok=True)
        paths["models_dir"] = str(models_dir)
    else:
        models_dir = Path(paths["models_dir"])

    # Log directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    options["paths"] = paths
    with open(output_dir / 'options.json', 'w') as fp:
        json.dump(options, fp, indent=4)

    save_useful_info(output_dir)

    train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        path_processed_data=path_processed_data,
        path_data_definition=path_data_definition,
        train_options=train_options,
        network_options=network_options,
    )


if __name__ == "__main__":
    train_zeggs()

# python .\main.py -o "../configs/configs.json" -n "test"

import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from anim.tquat import *


# ===============================================
#                   Decoder
# ===============================================
class Decoder(nn.Module):
    def __init__(
            self,
            pose_input_size,
            pose_output_size,
            speech_encoding_size,
            style_encoding_size,
            hidden_size,
            num_rnn_layers,
            rnn_cond="normal",
    ):
        super(Decoder, self).__init__()

        if rnn_cond == "normal":
            self.recurrent_decoder = RecurrentDecoderNormal(
                pose_input_size,
                speech_encoding_size,
                style_encoding_size,
                pose_output_size,
                hidden_size,
                num_rnn_layers,
            )
        elif rnn_cond == "film":
            self.recurrent_decoder = RecurrentDecoderFiLM(
                pose_input_size,
                speech_encoding_size,
                style_encoding_size,
                pose_output_size,
                hidden_size,
                num_rnn_layers,
            )

        self.cell_state_encoder = CellStateEncoder(
            pose_input_size + style_encoding_size, hidden_size, num_rnn_layers
        )

    def forward(
            self,
            Z_root_pos,
            Z_root_rot,
            Z_root_vel,
            Z_root_vrt,
            Z_lpos,
            Z_ltxy,
            Z_lvel,
            Z_lvrt,
            Z_gaze_pos,
            speech_encoding,
            style_encoding,
            parents,
            anim_input_mean,
            anim_input_std,
            anim_output_mean,
            anim_output_std,
            dt: float,
    ):

        batchsize = speech_encoding.shape[0]
        nframes = speech_encoding.shape[1]

        # Getting initial values from ground truth
        O_root_pos = [Z_root_pos]
        O_root_rot = [Z_root_rot]
        O_root_vel = [Z_root_vel]
        O_root_vrt = [Z_root_vrt]
        O_lpos = [Z_lpos]
        O_ltxy = [Z_ltxy]
        O_lvel = [Z_lvel]
        O_lvrt = [Z_lvrt]

        # Initialize the hidden state of decoder
        decoder_state = self.cell_state_encoder(
            vectorize_input(
                Z_root_pos,
                Z_root_rot,
                Z_root_vel,
                Z_root_vrt,
                Z_lpos,
                Z_ltxy,
                Z_lvel,
                Z_lvrt,
                Z_gaze_pos[:, 0],
                parents,
                anim_input_mean,
                anim_input_std,
            ),
            style_encoding[:, 0],
        )

        for i in range(1, nframes):
            # Prepare Input
            pose_encoding = vectorize_input(
                O_root_pos[-1],
                O_root_rot[-1],
                O_root_vel[-1],
                O_root_vrt[-1],
                O_lpos[-1],
                O_ltxy[-1],
                O_lvel[-1],
                O_lvrt[-1],
                Z_gaze_pos[:, i],
                parents,
                anim_input_mean,
                anim_input_std,
            )

            # Predict
            predicted, decoder_state = self.recurrent_decoder(
                pose_encoding, speech_encoding[:, i], style_encoding[:, i], decoder_state
            )

            # Integrate Prediction
            (
                P_root_pos,
                P_root_rot,
                P_root_vel,
                P_root_vrt,
                P_lpos,
                P_ltxy,
                P_lvel,
                P_lvrt,
            ) = devectorize_output(
                predicted,
                O_root_pos[-1],
                O_root_rot[-1],
                Z_lpos.shape[0],
                Z_lpos.shape[1],
                dt,
                anim_output_mean,
                anim_output_std,
            )

            # Append
            O_root_pos.append(P_root_pos)
            O_root_rot.append(P_root_rot)
            O_root_vel.append(P_root_vel)
            O_root_vrt.append(P_root_vrt)
            O_lpos.append(P_lpos)
            O_ltxy.append(P_ltxy)
            O_lvel.append(P_lvel)
            O_lvrt.append(P_lvrt)

        return (
            torch.cat([O[:, None] for O in O_root_pos], dim=1),
            torch.cat([O[:, None] for O in O_root_rot], dim=1),
            torch.cat([O[:, None] for O in O_root_vel], dim=1),
            torch.cat([O[:, None] for O in O_root_vrt], dim=1),
            torch.cat([O[:, None] for O in O_lpos], dim=1),
            torch.cat([O[:, None] for O in O_ltxy], dim=1),
            torch.cat([O[:, None] for O in O_lvel], dim=1),
            torch.cat([O[:, None] for O in O_lvrt], dim=1),
        )


class RecurrentDecoderNormal(nn.Module):
    def __init__(
            self, pose_input_size, speech_size, style_size, output_size, hidden_size, num_rnn_layers
    ):
        super(RecurrentDecoderNormal, self).__init__()

        all_input_size = pose_input_size + speech_size + style_size
        self.layer0 = nn.Linear(all_input_size, hidden_size)
        self.layer1 = nn.GRU(
            all_input_size + hidden_size, hidden_size, num_rnn_layers, batch_first=True
        )

        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, pose, speech, style, cell_state):
        hidden = F.elu(self.layer0(torch.cat([pose, speech, style], dim=-1)))
        cell_output, cell_state = self.layer1(
            torch.cat([hidden, pose, speech, style], dim=-1).unsqueeze(1), cell_state
        )
        output = self.layer2(cell_output.squeeze(1))
        return output, cell_state


class RecurrentDecoderFiLM(nn.Module):
    def __init__(
            self, pose_input_size, speech_size, style_size, output_size, hidden_size, num_rnn_layers
    ):
        super(RecurrentDecoderFiLM, self).__init__()

        self.hidden_size = hidden_size
        self.gammas_predictor = LinearNorm(
            style_size, hidden_size * 2, w_init_gain="linear"
        )
        self.betas_predictor = LinearNorm(
            style_size, hidden_size * 2, w_init_gain="linear"
        )

        all_input_size = pose_input_size + speech_size + style_size
        self.layer0 = nn.Linear(pose_input_size + speech_size, hidden_size)
        self.layer1 = nn.GRU(
            pose_input_size + speech_size + hidden_size,
            hidden_size,
            num_rnn_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, pose, speech, style, cell_state):
        gammas = self.gammas_predictor(style)
        gammas = gammas + 1
        betas = self.betas_predictor(style)

        hidden = F.elu(self.layer0(torch.cat([pose, speech], dim=-1)))
        hidden = hidden * gammas[:, : self.hidden_size] + betas[:, : self.hidden_size]
        cell_output, cell_state = self.layer1(
            torch.cat([hidden, pose, speech], dim=-1).unsqueeze(1), cell_state
        )
        hidden = F.elu(self.layer2(cell_output.squeeze(1)))
        hidden = hidden * gammas[:, self.hidden_size:] + betas[:, self.hidden_size:]
        output = self.layer3(hidden)
        return output, cell_state


class CellStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers):
        super(CellStateEncoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.layer0 = nn.Linear(input_size, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * num_rnn_layers)

    def forward(self, pose, style):
        hidden = F.elu(self.layer0(torch.cat([pose, style], dim=-1)))
        hidden = F.elu(self.layer1(hidden))
        output = self.layer2(hidden)

        return output.reshape(output.shape[0], self.num_rnn_layers, -1).swapaxes(0, 1).contiguous()


# ===============================================
#                   Speech Encoder
# ===============================================
class SpeechEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechEncoder, self).__init__()

        self.layer0 = nn.Conv1d(
            input_size, hidden_size, kernel_size=1, padding="same", padding_mode="replicate"
        )
        self.drop0 = nn.Dropout(p=0.2)

        self.layer1 = nn.Conv1d(
            hidden_size, output_size, kernel_size=31, padding="same", padding_mode="replicate"
        )
        self.drop1 = nn.Dropout(p=0.2)

        self.layer2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x = self.drop0(F.elu(self.layer0(x)))
        x = self.drop1(F.elu(self.layer1(x)))
        x = torch.swapaxes(x, 1, 2)
        x = F.elu(self.layer2(x))

        return x


# ===============================================
#                   Style Encoder
# ===============================================
class StyleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size, type="attn", use_vae=False):
        super(StyleEncoder, self).__init__()
        self.use_vae = use_vae
        self.style_embedding_size = style_embedding_size
        output_size = 2 * style_embedding_size if use_vae else style_embedding_size
        if type == "gru":
            self.encoder = StyleEncoderGRU(input_size, hidden_size, output_size)
        elif type == "attn":
            self.encoder = StyleEncoderAttn(input_size, hidden_size, output_size)

    def forward(self, input, temprature: float = 1.0):
        encoder_output = self.encoder(input)
        if self.use_vae:
            mu, logvar = (
                encoder_output[:, : self.style_embedding_size],
                encoder_output[:, self.style_embedding_size:],
            )

            # re-parameterization trick
            std = torch.exp(0.5 * logvar) / temprature
            eps = torch.randn_like(std)

            style_embedding = mu + eps * std
            return style_embedding, mu, logvar
        else:
            return encoder_output, None, None


class StyleEncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderGRU, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            # AvgPoolNorm1D(kernel_size=2),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )
        self.rnn_layer = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.projection_layer = LinearNorm(
            hidden_size * 2, style_embedding_size, w_init_gain="linear"
        )

    def forward(self, input):
        input = self.convs(input)
        output, _ = self.rnn_layer(input)
        style_embedding = self.projection_layer(output[:, -1])
        return style_embedding


class StyleEncoderAttn(nn.Module):
    """ Style Encoder Module:
        - Positional Encoding
        - Nf x FFT Blocks
        - Linear Projection Layer
    """

    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderAttn, self).__init__()

        # positional encoding
        self.pos_enc = PositionalEncoding(style_embedding_size)

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            ConvNorm1D(
                hidden_size,
                style_embedding_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(style_embedding_size),
            nn.Dropout(0.2),
        )
        # FFT blocks
        blocks = []
        for _ in range(1):
            blocks.append(FFTBlock(style_embedding_size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        """ Forward function of Prosody Encoder:
            frames_energy = (B, T_max)
            frames_pitch = (B, T_max)
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        """
        output_lengths = torch.as_tensor(
            len(input) * [input.shape[1]], device=input.device, dtype=torch.int32
        )
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1)).to(input.device)  # (B, T_max, hidden_embed_dim)
        # pass through convs
        outputs = self.convs(input)  # (B, T_max, hidden_embed_dim)

        # create mask
        mask = ~get_mask_from_lengths(output_lengths)  # (B, T_max)
        # add encodings and mask tensor
        outputs = outputs + pos  # (B, T_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        # pass through FFT blocks
        for _, block in enumerate(self.blocks):
            outputs = block(outputs, None, mask)  # (B, T_max, hidden_embed_dim)
        # average pooling on the whole time sequence
        style_embedding = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(
            1
        )  # (B, hidden_embed_dim)

        return style_embedding


# ===============================================
#                   Sub-modules
# ===============================================
class LinearNorm(nn.Module):
    """ Linear Norm Module:
        - Linear Layer
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Linear Norm
            x = (*, in_dim)
        """
        x = self.linear_layer(x)  # (*, out_dim)

        return x


class PositionalEncoding(nn.Module):
    """ Positional Encoding Module:
        - Sinusoidal Positional Embedding
    """

    def __init__(self, embed_dim, max_len=20000, timestep=10000.0):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-np.log(timestep) / self.embed_dim)
        )  # (embed_dim // 2, )
        self.pos_enc = torch.FloatTensor(max_len, self.embed_dim).zero_()  # (max_len, embed_dim)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        """ Forward function of Positional Encoding:
            x = (B, N) -- Long or Int tensor
        """
        # initialize tensor
        nb_frames_max = torch.max(torch.cumsum(x, dim=1))
        pos_emb = torch.FloatTensor(
            x.size(0), nb_frames_max, self.embed_dim
        ).zero_()  # (B, nb_frames_max, embed_dim)
        # pos_emb = pos_emb.cuda(x.device, non_blocking=True).float()  # (B, nb_frames_max, embed_dim)

        # TODO: Check if we can remove the for loops
        for line_idx in range(x.size(0)):
            pos_idx = []
            for column_idx in range(x.size(1)):
                idx = x[line_idx, column_idx]
                pos_idx.extend([i for i in range(idx)])
            emb = self.pos_enc[pos_idx]  # (nb_frames, embed_dim)
            pos_emb[line_idx, : emb.size(0), :] = emb

        return pos_emb


class FFTBlock(nn.Module):
    """ FFT Block Module:
        - Multi-Head Attention
        - Position Wise Convolutional Feed-Forward
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(FFTBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feed_forward = PositionWiseConvFF(hidden_size)

    def forward(self, x, film_params, mask):
        """ Forward function of FFT Block:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
            mask = (B, L_max)
        """
        # attend
        attn_outputs, _ = self.attention(
            x, x, x, key_padding_mask=mask
        )  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(
            mask.unsqueeze(2), 0
        )  # (B, L_max, hidden_embed_dim)
        # feed-forward pass
        outputs = self.feed_forward(attn_outputs, film_params)  # (B, L_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)

        return outputs


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention Module:
        - Multi-Head Attention
            A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin
            "Attention is all you need",
            in NeurIPS, 2017.
        - Dropout
        - Residual Connection 
        - Layer Normalization
    """

    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, 4, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """ Forward function of Multi-Head Attention:
            query = (B, L_max, hidden_embed_dim)
            key = (B, T_max, hidden_embed_dim)
            value = (B, T_max, hidden_embed_dim)
            key_padding_mask = (B, T_max) if not None
            attn_mask = (L_max, T_max) if not None
        """
        # compute multi-head attention
        # attn_outputs = (L_max, B, hidden_embed_dim)
        # attn_weights = (B, L_max, T_max)
        attn_outputs, attn_weights = self.multi_head_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        attn_outputs = attn_outputs.transpose(0, 1)  # (B, L_max, hidden_embed_dim)
        # apply dropout
        attn_outputs = self.dropout(attn_outputs)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        attn_outputs = self.layer_norm(attn_outputs + query)  # (B, L_max, hidden_embed_dim)

        return attn_outputs, attn_weights


class PositionWiseConvFF(nn.Module):
    """ Position Wise Convolutional Feed-Forward Module:
        - 2x Conv 1D with ReLU
        - Dropout
        - Residual Connection 
        - Layer Normalization
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(PositionWiseConvFF, self).__init__()
        self.convs = nn.Sequential(
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            ),
            nn.Dropout(0.1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, film_params):
        """ Forward function of PositionWiseConvFF:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
        """
        # pass through convs
        outputs = self.convs(x)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        outputs = self.layer_norm(outputs + x)  # (B, L_max, hidden_embed_dim)
        # add FiLM transformation
        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert nb_gammas == outputs.size(2)
            gammas = film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            betas = film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            outputs = gammas * outputs + betas  # (B, L_max, hidden_embed_dim)

        return outputs


class ConvNorm1D(nn.Module):
    """ Conv Norm 1D Module:
        - Conv 1D
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        """
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


class AvgPoolNorm1D(nn.Module):
    def __init__(
            self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        super(AvgPoolNorm1D, self).__init__()
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.avgpool1d(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


# ===============================================
#                   Funcs
# ===============================================
@torch.jit.script
def normalize(x, eps: float = 1e-8):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script
def vectorize_input(
        Z_root_pos,
        Z_root_rot,
        Z_root_vel,
        Z_root_vrt,
        Z_lpos,
        Z_ltxy,
        Z_lvel,
        Z_lvrt,
        Z_gaze_pos,
        parents,
        anim_input_mean,
        anim_input_std,
):
    batchsize = Z_lpos.shape[0]

    # Compute Local Gaze
    # Z_gaze_dir = quat_inv_mul_vec(Z_root_rot, normalize(Z_gaze_pos - Z_root_pos))
    Z_gaze_dir = quat_inv_mul_vec(Z_root_rot, Z_gaze_pos - Z_root_pos)

    # Flatten the autoregressive input
    pose_encoding = torch.cat(
        [
            Z_root_vel.reshape([batchsize, -1]),
            Z_root_vrt.reshape([batchsize, -1]),
            Z_lpos.reshape([batchsize, -1]),
            Z_ltxy.reshape([batchsize, -1]),
            Z_lvel.reshape([batchsize, -1]),
            Z_lvrt.reshape([batchsize, -1]),
            Z_gaze_dir.reshape([batchsize, -1]),
        ],
        dim=1,
    )

    # Normalize
    return (pose_encoding - anim_input_mean) / anim_input_std


@torch.jit.script
def devectorize_output(
        predicted,
        Z_root_pos,
        Z_root_rot,
        batchsize: int,
        njoints: int,
        dt: float,
        anim_output_mean,
        anim_output_std,
):
    # Denormalize
    predicted = (predicted * anim_output_std) + anim_output_mean

    # Extract predictions
    P_root_vel = predicted[:, 0:3]
    P_root_vrt = predicted[:, 3:6]
    P_lpos = predicted[:, 6 + njoints * 0: 6 + njoints * 3].reshape([batchsize, njoints, 3])
    P_ltxy = predicted[:, 6 + njoints * 3: 6 + njoints * 9].reshape([batchsize, njoints, 2, 3])
    P_lvel = predicted[:, 6 + njoints * 9: 6 + njoints * 12].reshape([batchsize, njoints, 3])
    P_lvrt = predicted[:, 6 + njoints * 12: 6 + njoints * 15].reshape([batchsize, njoints, 3])

    # Update pose state
    P_root_pos = quat_mul_vec(Z_root_rot, P_root_vel * dt) + Z_root_pos
    P_root_rot = quat_mul(quat_from_helical(quat_mul_vec(Z_root_rot, P_root_vrt * dt)), Z_root_rot)

    return (P_root_pos, P_root_rot, P_root_vel, P_root_vrt, P_lpos, P_ltxy, P_lvel, P_lvrt)


def generalized_logistic_function(x, center=0.0, B=1.0, A=0.0, K=1.0, C=1.0, Q=1.0, nu=1.0):
    """ Equation of the generalised logistic function
        https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param x:           abscissa point where logistic function needs to be evaluated
    :param center:      abscissa point corresponding to starting time
    :param B:           growth rate
    :param A:           lower asymptote
    :param K:           upper asymptote when C=1.
    :param C:           change upper asymptote value
    :param Q:           related to value at starting time abscissa point
    :param nu:          affects near which asymptote maximum growth occurs

    :return: value of logistic function at abscissa point
    """
    value = A + (K - A) / (C + Q * np.exp(-B * (x - center))) ** (1 / nu)
    return value


def compute_KL_div(mu, logvar, iteration):
    """ Compute KL divergence loss
        mu = (B, embed_dim)
        logvar = (B, embed_dim)
    """
    # compute KL divergence
    # see Appendix B from VAE paper:
    # D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR, 2014.

    kl_weight_center = 7500  # iteration at which weight of KL divergence loss is 0.5
    kl_weight_growth_rate = 0.005  # growth rate for weight of KL divergence loss
    kl_threshold = 2e-1  # KL weight threshold
    # kl_threshold = 1.0

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B, )
    kl_div = torch.mean(kl_div)

    # compute weight for KL cost annealing:
    # S.R. Bowman, L. Vilnis, O. Vinyals, A.M. Dai, R. Jozefowicz, S. Bengio,
    # "Generating Sentences from a Continuous Space", arXiv:1511.06349, 2016.
    kl_div_weight = generalized_logistic_function(
        iteration, center=kl_weight_center, B=kl_weight_growth_rate,
    )
    # apply weight threshold
    kl_div_weight = min(kl_div_weight, kl_threshold)
    return kl_div, kl_div_weight


def compute_kl_uni_gaus(q_params: Tuple, p_params: Tuple):
    mu_q, log_var_q = q_params
    mu_p, log_var_p = p_params

    kl = 0.5 * (log_var_p - log_var_q) + (log_var_q.exp() + (mu_q - mu_p) ** 2) / (2 * log_var_p.exp()) - 0.5 + 1e-8
    kl = torch.sum(kl, dim=-1)
    kl = torch.mean(kl)
    return kl


def get_mask_from_lengths(lengths):
    """ Create a masked tensor from given lengths

    :param lengths:     torch.tensor of size (B, ) -- lengths of each example

    :return mask: torch.tensor of size (B, max_length) -- the masked tensor
    """
    max_len = torch.max(lengths)
    # ids = torch.arange(0, max_len).cuda(lengths.device, non_blocking=True).long()
    ids = torch.arange(0, max_len).long().to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(lengths.device)
    return mask

import argparse
import json
import pathlib
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console

from anim import bvh
from anim import quat
from anim.txform import *
from audio.audio_files import read_wavfile
from data_pipeline import preprocess_animation
from data_pipeline import preprocess_audio
from helpers import split_by_ratio
from utils import write_bvh


def generate_gesture(
        audio_file,
        styles,
        network_path,
        data_path,
        results_path,
        style_encoding_type="example",
        blend_type="add",
        blend_ratio=[0.5, 0.5],
        file_name=None,
        first_pose=None,
        temperature=1.0,
        seed=1234,
        use_gpu=True,
        use_script=False,
):
    """Generate stylized gesture from raw audio and style example (ZEGGS)

    Args:
        audio_file ([type]): Path to audio file. If None the function does not generate geture and only outputs the style embedding
        styles ([type]): What styles to use.
        Multiple styles are given for blending or stitching styles.
            Style Encoding Type == "example":
                This is a list of tuples S, where each tuple S provides info for one style.
                    - S[0] is the path to the bvh example or the style embedding vec to be used directly
                    - S[1] is a list or tuple of size two defining the start and end frame to be used. None if style embedding is used directly
            Style Encoding Type == "label":
                - List of style labels (names)

        network_path ([type]): Path to the networks
        data_path ([type]): Path to the data directory containing needed processing information
        results_path ([type]): Path to result directory
        style_encoding_type (str, optional): How to encode the style. Either "example" or "label". Defaults to "example".
        blend_type (str, optional): Blending type, stitch (transitioning) or add (mixing). Defaults to "add".
        blend_ratio (list, optional): The proportion of blending. If blend type is "stitch", this is the proportion of the length.
                                      of the output for this style. If the blend type is "add" this is the interpolation weight
                                      Defaults to [0.5, 0.5].
        file_name ([type], optional): Output file name. If none the audio and example file names are used. Defaults to None.
        first_pose ([type], optional): The info required as the first pose. It can either be the path to the bvh file for using
                                       first pose or the animation dictionary extracted by loading a bvh file.
                                       If None, the pose from the last example is used (only used for example-based stylization.
                                       Defaults to None.
        temperature (float, optional): VAE temprature. This adjusts the amount of stochasticity. Defaults to 1.0.
        seed (int, optional): Random seed. Defaults to 1234.
        use_gpu (bool, optional): Use gpu or cpu. Defaults to True.
        use_script (bool, optional): Use torch script. Defaults to False.

    Returns:
        final_style_encoding: The final style embedding. If blend_type is "stitch", it is the style embedding for each frame.
                              If blend_type is "add", it is the interpolated style embedding vector
    """

    # Load details
    path_network_speech_encoder_weights = network_path / "speech_encoder.pt"
    path_network_decoder_weights = network_path / "decoder.pt"
    if style_encoding_type == "example":
        path_network_style_encoder_weights = network_path / "style_encoder.pt"
    path_stat_data = data_path / "stats.npz"
    path_data_definition = data_path / "data_definition.json"
    path_data_pipeline_conf = data_path / "data_pipeline_conf.json"
    if results_path is not None:
        results_path.mkdir(exist_ok=True)
    assert (audio_file is None) == (results_path is None)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Data pipeline conf (We must use the same processing configuration as the one in training)
    with open(path_data_pipeline_conf, "r") as f:
        data_pipeline_conf = json.load(f)
    data_pipeline_conf = DictConfig(data_pipeline_conf)

    # Animation static info (Skeleton, FPS, etc)
    with open(path_data_definition, "r") as f:
        details = json.load(f)

    njoints = len(details["bone_names"])
    nlabels = len(details["label_names"])
    label_names = details["label_names"]
    bone_names = details["bone_names"]
    parents = torch.as_tensor(details["parents"], dtype=torch.long, device=device)
    dt = details["dt"]

    # Load Stats (Mean and Std of input/output)

    stat_data = np.load(path_stat_data)
    audio_input_mean = torch.as_tensor(
        stat_data["audio_input_mean"], dtype=torch.float32, device=device
    )
    audio_input_std = torch.as_tensor(
        stat_data["audio_input_std"], dtype=torch.float32, device=device
    )
    anim_input_mean = torch.as_tensor(
        stat_data["anim_input_mean"], dtype=torch.float32, device=device
    )
    anim_input_std = torch.as_tensor(
        stat_data["anim_input_std"], dtype=torch.float32, device=device
    )
    anim_output_mean = torch.as_tensor(
        stat_data["anim_output_mean"], dtype=torch.float32, device=device
    )
    anim_output_std = torch.as_tensor(
        stat_data["anim_output_std"], dtype=torch.float32, device=device
    )

    # Load Networks
    network_speech_encoder = torch.load(path_network_speech_encoder_weights, map_location=device).to(device)
    network_speech_encoder.eval()

    network_decoder = torch.load(path_network_decoder_weights, map_location=device).to(device)
    network_decoder.eval()
    
    if style_encoding_type == "example":
        network_style_encoder = torch.load(path_network_style_encoder_weights, map_location=device).to(device)
        network_style_encoder.eval()

    if use_script:
        network_speech_encoder_script = torch.jit.script(network_speech_encoder)
        network_decoder_script = torch.jit.script(network_decoder)
        if style_encoding_type == "example":
            network_style_encoder_script = torch.jit.script(network_style_encoder)
    else:
        network_speech_encoder_script = network_speech_encoder
        network_decoder_script = network_decoder
        if style_encoding_type == "example":
            network_style_encoder_script = network_style_encoder

    network_speech_encoder_script.eval()
    network_decoder_script.eval()
    if style_encoding_type == "example":
        network_style_encoder_script.eval()

    with torch.no_grad():
        # If audio is None we only output the style encodings
        if audio_file is not None:
            # Load Audio

            _, audio_data = read_wavfile(
                audio_file,
                rescale=True,
                desired_fs=16000,
                desired_nb_channels=None,
                out_type="float32",
                logger=None,
            )

            n_frames = int(round(60.0 * (len(audio_data) / 16000)))

            audio_features = torch.as_tensor(
                preprocess_audio(
                    audio_data,
                    60,
                    n_frames,
                    data_pipeline_conf.audio_conf,
                    feature_type=data_pipeline_conf.audio_feature_type,
                ),
                device=device,
                dtype=torch.float32,
            )
            speech_encoding = network_speech_encoder_script(
                (audio_features[np.newaxis] - audio_input_mean) / audio_input_std
            )

        # Style Encoding
        style_encodings = []

        for style in styles:
            if style_encoding_type == "example":
                if isinstance(style[0], pathlib.WindowsPath) or isinstance(style[0], pathlib.PosixPath):
                    anim_name = Path(style[0]).stem
                    anim_data = bvh.load(style[0])

                    # Trimming if start/end frames are given
                    if style[1] is not None:
                        anim_data["rotations"] = anim_data["rotations"][
                                                 style[1][0]: style[1][1]
                                                 ]
                        anim_data["positions"] = anim_data["positions"][
                                                 style[1][0]: style[1][1]
                                                 ]
                    anim_fps = int(np.ceil(1 / anim_data["frametime"]))
                    assert anim_fps == 60

                    # Extracting features
                    (
                        root_pos,
                        root_rot,
                        root_vel,
                        root_vrt,
                        lpos,
                        lrot,
                        ltxy,
                        lvel,
                        lvrt,
                        cpos,
                        crot,
                        ctxy,
                        cvel,
                        cvrt,
                        gaze_pos,
                        gaze_dir,
                    ) = preprocess_animation(anim_data)

                    # convert to tensor
                    nframes = len(anim_data["rotations"])
                    root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=device)
                    root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=device)
                    root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
                    root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
                    lpos = torch.as_tensor(lpos, dtype=torch.float32, device=device)
                    ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=device)
                    lvel = torch.as_tensor(lvel, dtype=torch.float32, device=device)
                    lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=device)
                    gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=device)

                    S_root_vel = root_vel.reshape(nframes, -1)
                    S_root_vrt = root_vrt.reshape(nframes, -1)
                    S_lpos = lpos.reshape(nframes, -1)
                    S_ltxy = ltxy.reshape(nframes, -1)
                    S_lvel = lvel.reshape(nframes, -1)
                    S_lvrt = lvrt.reshape(nframes, -1)
                    example_feature_vec = torch.cat(
                        [
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            torch.zeros_like(S_root_vel),
                        ],
                        dim=1,
                    )
                    example_feature_vec = (example_feature_vec - anim_input_mean) / anim_input_std

                    style_encoding, _, _ = network_style_encoder_script(
                        example_feature_vec[np.newaxis], temperature
                    )
                    style_encodings.append(style_encoding)

                elif isinstance(style[0], np.ndarray):
                    anim_name = style[1]
                    style_embeddding = torch.as_tensor(
                        style[0], dtype=torch.float32, device=device
                    )[np.newaxis]
                    style_encodings.append(style_embeddding)
            elif style_encoding_type == "label":
                # get the index of style in label names
                style_index = label_names.index(style)
                style_embeddding = torch.zeros((1, nlabels), dtype=torch.float32, device=device)
                style_embeddding[0, style_index] = 1.0
                style_encodings.append(style_embeddding)
                assert first_pose is not None
            else:
                raise ValueError("Unknown style encoding type")

        if blend_type == "stitch":
            if len(style_encodings) > 1:
                if audio_file is None:
                    final_style_encoding = style_encodings
                else:
                    assert len(styles) == len(blend_ratio)
                    se = split_by_ratio(n_frames, blend_ratio)
                    V_root_pos = []
                    V_root_rot = []
                    V_lpos = []
                    V_ltxy = []
                    final_style_encoding = []
                    for i, style_encoding in enumerate(style_encodings):
                        final_style_encoding.append(
                            style_encoding.unsqueeze(1).repeat((1, se[i][-1] - se[i][0], 1))
                        )
                    final_style_encoding = torch.cat(final_style_encoding, dim=1)
            else:
                final_style_encoding = style_encodings[0]
        elif blend_type == "add":
            # style_encoding = torch.mean(torch.stack(style_encodings), dim=0)
            if len(style_encodings) > 1:
                assert len(style_encodings) == len(blend_ratio)
                final_style_encoding = torch.matmul(
                    torch.stack(style_encodings, dim=1).transpose(2, 1),
                    torch.tensor(blend_ratio, device=device),
                )
            else:
                final_style_encoding = style_encodings[0]

        if audio_file is not None:
            se = np.array_split(np.arange(n_frames), len(style_encodings))

            if first_pose is not None:
                if isinstance(first_pose, pathlib.WindowsPath) or isinstance(first_pose, pathlib.PosixPath):
                    anim_data = bvh.load(first_pose)
                elif isinstance(first_pose, dict):
                    anim_data = first_pose.copy()
                (
                    root_pos,
                    root_rot,
                    root_vel,
                    root_vrt,
                    lpos,
                    lrot,
                    ltxy,
                    lvel,
                    lvrt,
                    cpos,
                    crot,
                    ctxy,
                    cvel,
                    cvrt,
                    gaze_pos,
                    gaze_dir,
                ) = preprocess_animation(anim_data)

                root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=device)
                root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=device)
                root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
                root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
                lpos = torch.as_tensor(lpos, dtype=torch.float32, device=device)
                ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=device)
                lvel = torch.as_tensor(lvel, dtype=torch.float32, device=device)
                lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=device)
                gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=device)

            root_pos_0 = root_pos[0][np.newaxis]
            root_rot_0 = root_rot[0][np.newaxis]
            root_vel_0 = root_vel[0][np.newaxis]
            root_vrt_0 = root_vrt[0][np.newaxis]
            lpos_0 = lpos[0][np.newaxis]
            ltxy_0 = ltxy[0][np.newaxis]
            lvel_0 = lvel[0][np.newaxis]
            lvrt_0 = lvrt[0][np.newaxis]

            if final_style_encoding.dim() == 2:
                final_style_encoding = final_style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1))
            (
                V_root_pos,
                V_root_rot,
                V_root_vel,
                V_root_vrt,
                V_lpos,
                V_ltxy,
                V_lvel,
                V_lvrt,
            ) = network_decoder_script(
                root_pos_0,
                root_rot_0,
                root_vel_0,
                root_vrt_0,
                lpos_0,
                ltxy_0,
                lvel_0,
                lvrt_0,
                gaze_pos[0: 0 + 1].repeat_interleave(speech_encoding.shape[1], dim=0)[
                    np.newaxis
                ],
                speech_encoding,
                final_style_encoding,
                parents,
                anim_input_mean,
                anim_input_std,
                anim_output_mean,
                anim_output_std,
                dt,
            )

            V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())

            if file_name is None:
                file_name = f"audio_{audio_file.stem}_label_{anim_name}"
            try:
                write_bvh(
                    str(results_path / (file_name + ".bvh")),
                    V_root_pos[0].detach().cpu().numpy(),
                    V_root_rot[0].detach().cpu().numpy(),
                    V_lpos[0].detach().cpu().numpy(),
                    V_lrot[0],
                    parents=parents.detach().cpu().numpy(),
                    names=bone_names,
                    order="zyx",
                    dt=dt,
                    start_position=np.array([0, 0, 0]),
                    start_rotation=np.array([1, 0, 0, 0]),
                )
                copyfile(audio_file, str(results_path / (file_name + ".wav")))

            except (PermissionError, OSError) as e:
                print(e)
    return final_style_encoding


if __name__ == "__main__":

    # CLI for generating gesture from one pair of audio and style files or multiple pairs through a csv file
    # For full functionality, please use the generate_gesture function

    console = Console()

    # Setting parser
    parser = argparse.ArgumentParser(prog="ZEGGS", description="Generate samples by ZEGGS model")

    parser.add_argument(
        "-o",
        "--options",
        type=str,
        help="Options filename (generated during training)",
    )
    parser.add_argument('-p', '--results_path', type=str,
                        help="Results path. Default if 'results' directory in the folder containing networks",
                        nargs="?", const=None, required=False)
    parser.add_argument('-se', '--style_encoding_type', type=str,
                        help="Style encoding type either 'example' or 'label'", default="example", required=False)

    # 1. Generating gesture from a single pair of audio and style files
    parser.add_argument('-s', '--style', type=str, help="Path to style example file", required=False)
    parser.add_argument('-a', '--audio', type=str, help="Path to audio file", required=False)
    parser.add_argument('-n', '--file_name', type=str,
                        help="Output file name. If not given it will be automatically constructed", required=False)
    parser.add_argument('-fp', '--first_pose', type=str, help="First pose bvh file", default=None, required=False)
    parser.add_argument('-t', '--temperature', type=float,
                        help="VAE temprature. This adjusts the amount of stochasticity.", nargs="?", default=1.0,
                        required=False)
    parser.add_argument('-r', '--seed', type=int, help="Random seed", nargs="?", default=1234, required=False)
    parser.add_argument('-g', '--use_gpu', help="Use GPU (Default is using CPU)", action="store_true", required=False)
    parser.add_argument('-f', '--frames', type=int, help="Start and end frame of the style example to be used", nargs=2,
                        required=False)

    # 2. Generating gesture(s) from a csv file (some of the other arguments will be ignored)
    parser.add_argument('-c', '--csv', type=str,
                        help="CSV file containing information about pairs of audio/style and other parameters",
                        required=False)

    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = json.load(f)

    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]

    base_path = Path(paths["base_path"])
    data_path = base_path / paths["path_processed_data"]

    network_path = Path(paths["models_dir"])
    output_path = Path(paths["output_dir"])

    results_path = args.results_path
    if results_path is None:
        results_path = Path(output_path) / "results"

    style_encoding_type = args.style_encoding_type

    if args.csv is not None:
        console.print("Getting arguments from CSV file")
        df = pd.read_csv(args.csv)
        for index, row in df.iterrows():
            if not row["generate"]:
                continue

            with console.status(console.rule(f"Generating Gesture {index + 1}/{len(df)}")):
                row["results_path"] = results_path
                row["options"] = args.options
                base_path = Path(row["base_path"])
                frames = [int(x) for x in row["frames"].split(" ")] if isinstance(row["frames"], str) else None

                console.print("Arguments:")
                console.print(row.to_string(index=True))
                style = [(base_path / Path(row["style"]), frames)] if style_encoding_type == "example" else [
                    row["style"]]
                generate_gesture(
                    audio_file=base_path / Path(row["audio"]),
                    styles=style,
                    network_path=network_path,
                    data_path=data_path,
                    results_path=results_path,
                    style_encoding_type=style_encoding_type,
                    file_name=row["file_name"],
                    first_pose=base_path / Path(row["first_pose"]),
                    temperature=row["temperature"],
                    seed=row["seed"],
                    use_gpu=row["use_gpu"]
                )
    else:
        with console.status(console.rule("Generating Gesture")):
            console.print("Arguments:")
            df = pd.DataFrame([vars(args)])
            console.print(df.iloc[0].to_string(index=True))
            file_name = args.file_name
            style = [(Path(args.style), args.frames)] if style_encoding_type == "example" else [args.style]
            generate_gesture(
                audio_file=Path(args.audio),
                styles=style,
                network_path=network_path,
                data_path=data_path,
                results_path=results_path,
                style_encoding_type=style_encoding_type,
                file_name=args.file_name,
                first_pose=args.first_pose,
                temperature=args.temperature,
                seed=args.seed,
                use_gpu=args.use_gpu
            )

import numpy as np
from scipy import interpolate

from anim import bvh, quat


def change_bvh(filename, savename, order=None, fps=None, pace=1.0, center=False):
    anim_data = bvh.load(filename)
    output = anim_data.copy()

    if order is not None:
        output["order"] = order
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
    if pace is not None or fps is not None:
        if fps is None:
            fps = 1.0 / anim_data["frametime"]
        positions = anim_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(
            0, nframes - 1, int(pace * (nframes * (fps * anim_data["frametime"]) - 1))
        )
        output["positions"] = interpolate.griddata(original_times, output["positions"].reshape([nframes, -1]),
                                                   sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = interpolate.griddata(original_times, rotations.reshape([nframes, -1]),
                                         sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        rotations = quat.normalize(rotations)
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
        output["frametime"] = 1.0 / fps

    if center:
        lrot = quat.from_euler(np.radians(output["rotations"]), output["order"])
        offset_pos = output["positions"][0:1, 0:1].copy() * np.array([1, 0, 1])
        offset_rot = lrot[0:1, 0:1].copy() * np.array([1, 0, 1, 0])

        root_pos = quat.mul_vec(quat.inv(offset_rot), output["positions"][:, 0:1] - offset_pos)
        output["positions"][:, 0:1] = quat.mul_vec(quat.inv(offset_rot),
                                                   output["positions"][:, 0:1] - offset_pos)
        output["rotations"][:, 0:1] = np.degrees(
            quat.to_euler(quat.mul(quat.inv(offset_rot), lrot[:, 0:1]), order=output["order"]))
    bvh.save(savename, output)


def write_bvh(
        filename,
        V_root_pos,
        V_root_rot,
        V_lpos,
        V_lrot,
        parents,
        names,
        order,
        dt,
        start_position=None,
        start_rotation=None,
):
    if start_position is not None and start_rotation is not None:
        offset_pos = V_root_pos[0:1].copy()
        offset_rot = V_root_rot[0:1].copy()

        V_root_pos = quat.mul_vec(quat.inv(offset_rot), V_root_pos - offset_pos)
        V_root_rot = quat.mul(quat.inv(offset_rot), V_root_rot)
        V_root_pos = (
                quat.mul_vec(start_rotation[np.newaxis], V_root_pos) + start_position[np.newaxis]
        )
        V_root_rot = quat.mul(start_rotation[np.newaxis], V_root_rot)

    V_lpos = V_lpos.copy()
    V_lrot = V_lrot.copy()
    V_lpos[:, 0] = quat.mul_vec(V_root_rot, V_lpos[:, 0]) + V_root_pos
    V_lrot[:, 0] = quat.mul(V_root_rot, V_lrot[:, 0])

    bvh.save(
        filename,
        dict(
            order=order,
            offsets=V_lpos[0],
            names=names,
            frametime=dt,
            parents=parents,
            positions=V_lpos,
            rotations=np.degrees(quat.to_euler(V_lrot, order=order)),
        ),
    )

import sys, os, logging
import pyfbsdk

# Customs
# MoBu env
sys.path.append("C:/Users/sghorbani/Anaconda3/envs/mobu2/Lib/site-packages/")


import numpy as np

logging.basicConfig(
    filename="compile_results.log",
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logging.getLogger("").addHandler(console)


def bvh2fbx(animation_file, output_file, template_file, sound_file=None):

    pyfbsdk.FBApplication().FileNew()
    logging.info("Loading %s..." % str(template_file))

    if not pyfbsdk.FBApplication().FileOpen(str(template_file)):
        raise IOError("Could not open file: {}".format(str(template_file)))

    if sound_file is not None:
        # Load Audio
        logging.info("Loading %s..." % str(sound_file))
        audio = pyfbsdk.FBAudioClip(sound_file)
        if audio is None:
            raise IOError("Could not open file: {}".format(str(sound_file)))

        # Rescale Timespan
        pyfbsdk.FBSystem().CurrentTake.LocalTimeSpan = pyfbsdk.FBTimeSpan(
            pyfbsdk.FBTime(0), audio.Duration
        )

    # Set FPS
    pyfbsdk.FBPlayerControl().SetTransportFps(pyfbsdk.FBTimeMode.kFBTimeMode60Frames)
    pyfbsdk.FBPlayerControl().SnapMode = (
        pyfbsdk.FBTransportSnapMode.kFBTransportSnapModeSnapOnFrames
    )

    # Load BVH
    if not pyfbsdk.FBApplication().FileImport(animation_file, True):
        raise IOError("Could not open file: {}".format(str(animation_file)))

    # Save FBX
    pyfbsdk.FBApplication().FileSave(output_file)


if True:
    try:

        logging.info("======")
        logging.info("BVH2FBX")
        logging.info("======")

        results_path = "./Rendered"
        template_file = "./LaForgeFemale.fbx"

        # Characterizing all bvh files
        for animation_file in [f for f in os.listdir(results_path) if f.endswith(".bvh")]:
            sound_file = animation_file.replace(".bvh", ".wav")
            sound_file = results_path + "/" + sound_file
            if not os.path.exists(sound_file):
                sound_file = None
            bvh2fbx(
                results_path + "/" + animation_file,
                results_path + "/" + animation_file.replace(".bvh", ".fbx"),
                template_file,
                sound_file
            )

        pyfbsdk.FBApplication().FileExit()

    except Exception as e:
        logging.exception("FAILED:")
        raise e


""" S2G Training """
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anim import quat
from anim.tquat import *
from anim.txform import *
from dataset import SGDataset
from helpers import flatten_dict
from helpers import progress
from helpers import save_useful_info
from modules import Decoder
from modules import SpeechEncoder
from modules import StyleEncoder
from modules import compute_KL_div
from modules import normalize
from optimizers import RAdam
from utils import write_bvh


def train(
        models_dir,
        logs_dir,
        path_processed_data,
        path_data_definition,
        train_options,
        network_options,
):
    # ===============================================
    #    Getting/Setting Training/Network Configs
    # ===============================================
    np.random.seed(train_options["seed"])
    torch.manual_seed(train_options["seed"])
    torch.set_num_threads(train_options["thread_count"])
    use_gpu = train_options["use_gpu"] and torch.cuda.is_available()
    use_script = train_options["use_script"]

    if use_gpu:
        print("Using GPU!")
    else:
        print("Using CPU!")
    device = torch.device("cuda:0" if use_gpu else "cpu")
    window = train_options["window"]
    niterations = train_options["niterations"]
    batchsize = train_options["batchsize"]
    style_encoder_opts = network_options["style_encoder"]
    speech_encoder_opts = network_options["speech_encoder"]
    decoder_opts = network_options["decoder"]

    # ===============================================
    #                   Load Details
    # ===============================================
    with open(path_data_definition, "r") as f:
        details = json.load(f)

    nlabels = len(details["label_names"])
    bone_names = details["bone_names"]
    parents = torch.LongTensor(details["parents"])
    dt = details["dt"]

    # ===============================================
    #                   Load Data
    # ===============================================
    ds = SGDataset(
        path_data_definition,
        path_processed_data,
        window,
        style_encoding_type=train_options["style_encoding_type"],
        example_window_length=style_encoder_opts["example_length"],
    )
    # Workaround: The number of workers should be 0 so that the example length can be changed dynamically
    dl = DataLoader(ds, drop_last=True, batch_size=batchsize, shuffle=True, num_workers=0)

    dimensions = ds.get_shapes()

    (
        audio_input_mean,
        audio_input_std,
        anim_input_mean,
        anim_input_std,
        anim_output_mean,
        anim_output_std,
    ) = ds.get_means_stds(device)

    # ===============================================
    #             Load or Resume Networks
    # ===============================================
    style_encoding_type = train_options["style_encoding_type"]
    if style_encoding_type == "label":
        style_encoding_size = nlabels
    elif style_encoding_type == "example":
        style_encoding_size = style_encoder_opts["style_encoding_size"]

    path_network_speech_encoder_weights = models_dir / "speech_encoder.pt"
    path_network_decoder_weights = models_dir / "decoder.pt"
    path_network_style_encoder_weights = models_dir / "style_encoder.pt"
    path_checkpoints = models_dir / "checkpoints.pt"

    if (
            train_options["resume"]
            and os.path.exists(path_network_speech_encoder_weights)
            and os.path.exists(path_network_decoder_weights)
            and os.path.exists(path_checkpoints)
    ):
        network_speech_encoder = torch.load(path_network_speech_encoder_weights).to(device)
        network_decoder = torch.load(path_network_decoder_weights).to(device)
        network_style_encoder = torch.load(path_network_style_encoder_weights).to(device)

    else:
        network_speech_encoder = SpeechEncoder(
            dimensions["num_audio_features"],
            speech_encoder_opts["nhidden"],
            speech_encoder_opts["speech_encoding_size"],
        ).to(device)

        network_decoder = Decoder(
            pose_input_size=dimensions["pose_input_size"],
            pose_output_size=dimensions["pose_output_size"],
            speech_encoding_size=speech_encoder_opts["speech_encoding_size"],
            style_encoding_size=style_encoding_size,
            hidden_size=decoder_opts["nhidden"],
            num_rnn_layers=2,
        ).to(device)
        if style_encoding_type == "example":
            network_style_encoder = StyleEncoder(
                dimensions["pose_input_size"],
                style_encoder_opts["nhidden"],
                style_encoding_size,
                type=style_encoder_opts["type"],
                use_vae=style_encoder_opts["use_vae"],
            ).to(device)

    if use_script:
        network_speech_encoder_script = torch.jit.script(network_speech_encoder)
        network_decoder_script = torch.jit.script(network_decoder)
        if style_encoding_type == "example":
            network_style_encoder_script = torch.jit.script(network_style_encoder)
    else:
        network_speech_encoder_script = network_speech_encoder
        network_decoder_script = network_decoder
        if style_encoding_type == "example":
            network_style_encoder_script = network_style_encoder

    # ===============================================
    #                   Optimizer
    # ===============================================
    all_parameters = (
            list(network_speech_encoder.parameters())
            + list(network_decoder.parameters())
            + (list(network_style_encoder.parameters() if style_encoding_type == "example" else []))
    )
    optimizer = RAdam(all_parameters, lr=train_options["learning_rate"], eps=train_options["eps"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, train_options["learning_rate_decay"]
    )

    if train_options["resume"]:
        checkpoints = torch.load(path_checkpoints)
        iteration = checkpoints["iteration"]
        epoch = checkpoints["epoch"]
        loss = checkpoints["loss"]
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    else:
        iteration = 0
        epoch = 0

    # ===============================================
    #             Setting Log Directories
    # ===============================================
    samples_dir = logs_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    if train_options["use_tensorboard"]:
        tb_dir = logs_dir / "tb"
        tb_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(tb_dir, flush_secs=10)
        hparams = flatten_dict(network_options)
        hparams.update(flatten_dict(train_options))
        writer.add_hparams(hparams, {"No Metric": 0.0})

    # ===============================================
    #                 Begin Training
    # ===============================================
    while iteration < (1000 * niterations):

        start_time = datetime.datetime.now()

        for batch_index, batch in enumerate(dl):
            network_speech_encoder.train()
            network_decoder.train()
            if style_encoding_type == "example":
                network_style_encoder.train()
            (
                W_audio_features,
                W_root_pos,
                W_root_rot,
                W_root_vel,
                W_root_vrt,
                W_lpos,
                W_ltxy,
                W_lvel,
                W_lvrt,
                W_gaze_pos,
                WStyle,
            ) = batch

            W_audio_features = W_audio_features.to(device)
            W_root_pos = W_root_pos.to(device)
            W_root_rot = W_root_rot.to(device)
            W_root_vel = W_root_vel.to(device)
            W_root_vrt = W_root_vrt.to(device)
            W_lpos = W_lpos.to(device)
            W_ltxy = W_ltxy.to(device)
            W_lvel = W_lvel.to(device)
            W_lvrt = W_lvrt.to(device)
            W_gaze_pos = W_gaze_pos.to(device)
            WStyle = WStyle.to(device)

            # Dynamically changing example length for the next iteration
            ds.example_window_length = 2 * random.randint(style_encoder_opts["example_length"] // 2,
                                                          style_encoder_opts["example_length"])

            # Speech Encoder
            speech_encoding = network_speech_encoder_script(
                (W_audio_features - audio_input_mean) / audio_input_std
            )

            # Style Encoder
            mu, logvar = None, None
            if style_encoding_type == "example":
                WStyle = (WStyle - anim_input_mean) / anim_input_std
                style_encoding, mu, logvar = network_style_encoder_script(
                    WStyle.to(device=device)
                )
            else:
                style_encoding = WStyle

            # Gesture Generator
            (
                O_root_pos,
                O_root_rot,
                O_root_vel,
                O_root_vrt,
                O_lpos,
                O_ltxy,
                O_lvel,
                O_lvrt,
            ) = network_decoder_script(
                W_root_pos[:, 0],
                W_root_rot[:, 0],
                W_root_vel[:, 0],
                W_root_vrt[:, 0],
                W_lpos[:, 0],
                W_ltxy[:, 0],
                W_lvel[:, 0],
                W_lvrt[:, 0],
                W_gaze_pos,
                speech_encoding,
                style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                parents,
                anim_input_mean,
                anim_input_std,
                anim_output_mean,
                anim_output_std,
                dt,
            )

            # Compute Character/World Space
            W_lmat = xform_orthogonalize_from_xy(W_ltxy)
            O_lmat = xform_orthogonalize_from_xy(O_ltxy)

            ## Root Velocities to World Space
            O_root_vel_1_ = quat_mul_vec(O_root_rot[:, :-1], O_root_vel[:, 1:])
            O_root_vrt_1_ = quat_mul_vec(O_root_rot[:, :-1], O_root_vrt[:, 1:])
            O_root_vel_0 = quat_mul_vec(O_root_rot[:, 0:1], O_root_vel[:, 0:1])
            O_root_vrt_0 = quat_mul_vec(O_root_rot[:, 0:1], O_root_vrt[:, 0:1])
            O_root_vel = torch.cat((O_root_vel_0, O_root_vel_1_), dim=1)
            O_root_vrt = torch.cat((O_root_vrt_0, O_root_vrt_1_), dim=1)

            W_root_vel_1_ = quat_mul_vec(W_root_rot[:, :-1], W_root_vel[:, 1:])
            W_root_vrt_1_ = quat_mul_vec(W_root_rot[:, :-1], W_root_vrt[:, 1:])
            W_root_vel_0 = quat_mul_vec(W_root_rot[:, 0:1], W_root_vel[:, 0:1])
            W_root_vrt_0 = quat_mul_vec(W_root_rot[:, 0:1], W_root_vrt[:, 0:1])
            W_root_vel = torch.cat((W_root_vel_0, W_root_vel_1_), dim=1)
            W_root_vrt = torch.cat((W_root_vrt_0, W_root_vrt_1_), dim=1)

            ## Update First Joint
            O_lpos_0 = quat_mul_vec(O_root_rot, O_lpos[:, :, 0]) + O_root_pos
            O_lmat_0 = torch.matmul(quat_to_xform(O_root_rot), O_lmat[:, :, 0])
            O_lvel_0 = (
                    O_root_vel
                    + quat_mul_vec(O_root_rot, O_lvel[:, :, 0])
                    + torch.cross(O_root_vrt, quat_mul_vec(O_root_rot, O_lpos[:, :, 0]))
            )
            O_lvrt_0 = O_root_vrt + quat_mul_vec(O_root_rot, O_lvrt[:, :, 0])

            O_lpos = torch.cat((O_lpos_0.unsqueeze(2), O_lpos[:, :, 1:]), dim=2)
            O_lmat = torch.cat((O_lmat_0.unsqueeze(2), O_lmat[:, :, 1:]), dim=2)
            O_lvel = torch.cat((O_lvel_0.unsqueeze(2), O_lvel[:, :, 1:]), dim=2)
            O_lvrt = torch.cat((O_lvrt_0.unsqueeze(2), O_lvrt[:, :, 1:]), dim=2)

            W_lpos_0 = quat_mul_vec(W_root_rot, W_lpos[:, :, 0]) + W_root_pos
            W_lmat_0 = torch.matmul(quat_to_xform(W_root_rot), W_lmat[:, :, 0])
            W_lvel_0 = (
                    W_root_vel
                    + quat_mul_vec(W_root_rot, W_lvel[:, :, 0])
                    + torch.cross(W_root_vrt, quat_mul_vec(W_root_rot, W_lpos[:, :, 0]))
            )
            W_lvrt_0 = W_root_vrt + quat_mul_vec(W_root_rot, W_lvrt[:, :, 0])

            W_lpos = torch.cat((W_lpos_0.unsqueeze(2), W_lpos[:, :, 1:]), dim=2)
            W_lmat = torch.cat((W_lmat_0.unsqueeze(2), W_lmat[:, :, 1:]), dim=2)
            W_lvel = torch.cat((W_lvel_0.unsqueeze(2), W_lvel[:, :, 1:]), dim=2)
            W_lvrt = torch.cat((W_lvrt_0.unsqueeze(2), W_lvrt[:, :, 1:]), dim=2)

            # Fk to Character or World Space
            W_cmat, W_cpos, W_cvrt, W_cvel = xform_fk_vel(
                W_lmat, W_lpos, W_lvrt, W_lvel, parents
            )
            O_cmat, O_cpos, O_cvrt, O_cvel = xform_fk_vel(
                O_lmat, O_lpos, O_lvrt, O_lvel, parents
            )

            O_root_mat = quat_to_xform(O_root_rot)
            W_root_mat = quat_to_xform(W_root_rot)

            # Compute Gaze Dirs
            W_gaze_dir = quat_inv_mul_vec(W_root_rot, normalize(W_gaze_pos - W_root_pos))
            O_gaze_dir = quat_inv_mul_vec(O_root_rot, normalize(W_gaze_pos - O_root_pos))

            # Compute Losses
            loss_root_pos = torch.mean(torch.abs(0.1 * (O_root_pos - W_root_pos)))
            loss_root_rot = torch.mean(torch.abs(10.0 * (O_root_mat - W_root_mat)))
            loss_root_vel = torch.mean(torch.abs(0.1 * (O_root_vel - W_root_vel)))
            loss_root_vrt = torch.mean(torch.abs(5.0 * (O_root_vrt - W_root_vrt)))

            loss_lpos = torch.mean(torch.abs(15.0 * (O_lpos - W_lpos)))
            loss_lrot = torch.mean(torch.abs(15.0 * (O_ltxy - W_ltxy)))
            loss_lvel = torch.mean(torch.abs(10.0 * (O_lvel - W_lvel)))
            loss_lvrt = torch.mean(torch.abs(7.0 * (O_lvrt - W_lvrt)))

            loss_cpos = torch.mean(torch.abs(0.1 * (O_cpos - W_cpos)))
            loss_crot = torch.mean(torch.abs(3.0 * (O_cmat - W_cmat)))
            loss_cvel = torch.mean(torch.abs(0.06 * (O_cvel - W_cvel)))
            loss_cvrt = torch.mean(torch.abs(1.25 * (O_cvrt - W_cvrt)))

            loss_ldvl = torch.mean(
                torch.abs(
                    7.0
                    * (
                            (O_lpos[:, 1:] - O_lpos[:, :-1]) / dt
                            - (W_lpos[:, 1:] - W_lpos[:, :-1]) / dt
                    )
                )
            )

            loss_ldvt = torch.mean(
                torch.abs(
                    8.0
                    * (
                            (O_ltxy[:, 1:] - O_ltxy[:, :-1]) / dt
                            - (W_ltxy[:, 1:] - W_ltxy[:, :-1]) / dt
                    )
                )
            )

            loss_cdvl = torch.mean(
                torch.abs(
                    0.06
                    * (
                            (O_cpos[:, 1:] - O_cpos[:, :-1]) / dt
                            - (W_cpos[:, 1:] - W_cpos[:, :-1]) / dt
                    )
                )
            )

            loss_cdvt = torch.mean(
                torch.abs(
                    1.25
                    * (
                            (O_cmat[:, 1:] - O_cmat[:, :-1]) / dt
                            - (W_cmat[:, 1:] - W_cmat[:, :-1]) / dt
                    )
                )
            )

            loss_gaze = torch.mean(torch.abs(10.0 * (O_gaze_dir - W_gaze_dir)))

            loss_kl_div = 0.0
            if mu is not None and logvar is not None:
                kl_div, kl_div_weight = compute_KL_div(mu, logvar, iteration)
                loss_kl_div = kl_div_weight * torch.mean(kl_div)

            loss = (
                           +loss_root_pos
                           + loss_root_rot
                           + loss_root_vel
                           + loss_root_vrt
                           + loss_lpos
                           + loss_lrot
                           + loss_lvel
                           + loss_lvrt
                           + loss_cpos
                           + loss_crot
                           + loss_cvel
                           + loss_cvrt
                           + loss_ldvl
                           + loss_ldvt
                           + loss_cdvl
                           + loss_cdvt
                           + loss_gaze
                           + loss_kl_div
                   ) / 18.0

            # Backward
            loss.backward()
            optimizer.step()

            # Zero Gradients
            optimizer.zero_grad()

            losses = loss.detach().item()
            if (iteration + 1) % 1000 == 0:
                scheduler.step()

            # ===================================================
            #           Logging, Generating Samples
            # ===================================================
            if train_options["use_tensorboard"]:
                writer.add_scalar("losses/total_loss", loss, iteration)

                writer.add_scalars(
                    "losses/losses",
                    {
                        "loss_root_pos": loss_root_pos,
                        "loss_root_rot": loss_root_rot,
                        "loss_root_vel": loss_root_vel,
                        "loss_root_vrt": loss_root_vrt,
                        "loss_lpos": loss_lpos,
                        "loss_lrot": loss_lrot,
                        "loss_lvel": loss_lvel,
                        "loss_lvrt": loss_lvrt,
                        "loss_cpos": loss_cpos,
                        "loss_crot": loss_crot,
                        "loss_cvel": loss_cvel,
                        "loss_cvrt": loss_cvrt,
                        "loss_ldvl": loss_ldvl,
                        "loss_ldvt": loss_ldvt,
                        "loss_cdvl": loss_cdvl,
                        "loss_cdvt": loss_cdvt,
                        "loss_gaze": loss_gaze,
                        "loss_kl_div": loss_kl_div,
                    },
                    iteration,
                )

            if (iteration + 1) % 1 == 0:
                sys.stdout.write(
                    "\r"
                    + progress(
                        epoch,
                        iteration,
                        batch_index,
                        np.mean(losses),
                        (len(ds) // batchsize),
                        start_time,
                    )
                )
            if iteration % train_options["generate_samples_step"] == 0:
                sys.stdout.write(
                    "\r|                             Saving Networks...                                  |"
                )

                torch.save(network_speech_encoder, path_network_speech_encoder_weights)
                torch.save(network_decoder, path_network_decoder_weights)
                if style_encoding_type == "example":
                    torch.save(network_style_encoder, path_network_style_encoder_weights)
                torch.save({
                    'iteration': iteration,
                    "epoch": epoch,
                    'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, models_dir / "checkpoints.pt")

                current_models_dir = models_dir / str(iteration)
                current_models_dir.mkdir(exist_ok=True)

                path_network_speech_encoder_weights_current = current_models_dir / "speech_encoder.pt"
                path_network_decoder_weights_current = current_models_dir / "decoder.pt"
                path_network_style_encoder_weights_current = current_models_dir / "style_encoder.pt"

                torch.save(network_speech_encoder, path_network_speech_encoder_weights_current)
                torch.save(network_decoder, path_network_decoder_weights_current)
                if style_encoding_type == "example":
                    torch.save(network_style_encoder, path_network_style_encoder_weights_current)
                torch.save({
                    'iteration': iteration,
                    "epoch": epoch,
                    'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, current_models_dir / "checkpoints.pt")

                with torch.no_grad():
                    network_speech_encoder.eval()
                    network_decoder.eval()
                    if style_encoding_type == "example":
                        network_style_encoder.eval()
                    sys.stdout.write(
                        "\r|                           Generating Animation...                               |"
                    )

                    # Write training animation
                    for i in range(3):
                        (
                            S_audio_features,
                            S_root_pos,
                            S_root_rot,
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            S_gaze_pos,
                            label,
                            se,
                            range_index,
                        ) = ds.get_sample("train", 30)

                        speech_encoding = network_speech_encoder_script(
                            (S_audio_features.to(device=device) - audio_input_mean)
                            / audio_input_std
                        )

                        if style_encoding_type == "example":
                            example = ds.get_example(se, se, ds.example_window_length)
                            example = (example.to(device=device) - anim_input_mean) / anim_input_std
                            style_encoding, _, _ = network_style_encoder_script(example[np.newaxis])
                        else:
                            style_encoding = np.zeros([nlabels])
                            style_encoding[label] = 1.0
                            style_encoding = torch.as_tensor(
                                style_encoding, dtype=torch.float32, device=device
                            )[np.newaxis]

                        (
                            V_root_pos,
                            V_root_rot,
                            _,
                            _,
                            V_lpos,
                            V_ltxy,
                            _,
                            _,
                        ) = network_decoder_script(
                            S_root_pos[:, 0].to(device=device),
                            S_root_rot[:, 0].to(device=device),
                            S_root_vel[:, 0].to(device=device),
                            S_root_vrt[:, 0].to(device=device),
                            S_lpos[:, 0].to(device=device),
                            S_ltxy[:, 0].to(device=device),
                            S_lvel[:, 0].to(device=device),
                            S_lvrt[:, 0].to(device=device),
                            S_gaze_pos.to(device=device),
                            speech_encoding,
                            style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                            parents,
                            anim_input_mean,
                            anim_input_std,
                            anim_output_mean,
                            anim_output_std,
                            dt,
                        )

                        S_lrot = quat.from_xform(xform_orthogonalize_from_xy(S_ltxy).cpu().numpy())
                        V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).cpu().numpy())

                        try:
                            current_label = details["label_names"][label]
                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_train_ground_{i}_{current_label}.bvh"
                                    )
                                ),
                                S_root_pos[0].cpu().numpy(),
                                S_root_rot[0].cpu().numpy(),
                                S_lpos[0].cpu().numpy(),
                                S_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_train_predict_{i}_{current_label}.bvh"
                                    )
                                ),
                                V_root_pos[0].cpu().numpy(),
                                V_root_rot[0].cpu().numpy(),
                                V_lpos[0].cpu().numpy(),
                                V_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                        except (PermissionError, OSError) as e:
                            print(e)

                    # Write validation animation

                    for i in range(3):
                        (
                            S_audio_features,
                            S_root_pos,
                            S_root_rot,
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            S_gaze_pos,
                            label,
                            se,
                            range_index,
                        ) = ds.get_sample("valid", 30)

                        speech_encoding = network_speech_encoder_script(
                            (S_audio_features.to(device=device) - audio_input_mean)
                            / audio_input_std
                        )

                        if style_encoding_type == "example":
                            example = ds.get_example(se, se, ds.example_window_length)
                            example = (example.to(device=device) - anim_input_mean) / anim_input_std
                            style_encoding, _, _ = network_style_encoder_script(example[np.newaxis])
                        else:
                            style_encoding = np.zeros([nlabels])
                            style_encoding[label] = 1.0
                            style_encoding = torch.as_tensor(
                                style_encoding, dtype=torch.float32, device=device
                            )[np.newaxis]

                        (
                            V_root_pos,
                            V_root_rot,
                            _,
                            _,
                            V_lpos,
                            V_ltxy,
                            _,
                            _,
                        ) = network_decoder_script(
                            S_root_pos[:, 0].to(device=device),
                            S_root_rot[:, 0].to(device=device),
                            S_root_vel[:, 0].to(device=device),
                            S_root_vrt[:, 0].to(device=device),
                            S_lpos[:, 0].to(device=device),
                            S_ltxy[:, 0].to(device=device),
                            S_lvel[:, 0].to(device=device),
                            S_lvrt[:, 0].to(device=device),
                            S_gaze_pos.to(device=device),
                            speech_encoding,
                            style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                            parents,
                            anim_input_mean,
                            anim_input_std,
                            anim_output_mean,
                            anim_output_std,
                            dt,
                        )

                        S_lrot = quat.from_xform(xform_orthogonalize_from_xy(S_ltxy).cpu().numpy())
                        V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).cpu().numpy())

                        try:
                            current_label = details["label_names"][label]
                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_valid_ground_{i}_{current_label}.bvh"
                                    )
                                ),
                                S_root_pos[0].cpu().numpy(),
                                S_root_rot[0].cpu().numpy(),
                                S_lpos[0].cpu().numpy(),
                                S_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_valid_predict_{i}_{current_label}.bvh"
                                    )
                                ),
                                V_root_pos[0].cpu().numpy(),
                                V_root_rot[0].cpu().numpy(),
                                V_lpos[0].cpu().numpy(),
                                V_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                        except (PermissionError, OSError) as e:
                            print(e)

            iteration += 1
        sys.stdout.write("\n")

        epoch += 1
    print("Done!")


if __name__ == "__main__":

    # For debugging
    options = "../configs/configs_v2.json"
    with open(options, "r") as f:
        options = json.load(f)

    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]

    base_path = Path(paths["base_path"])
    path_processed_data = base_path / paths["path_processed_data"] / "processed_data.npz"
    path_data_definition = base_path / paths["path_processed_data"] / "data_definition.json"

    # Output directory
    if paths["output_dir"] is None:
        output_dir = (base_path / "outputs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_dir.mkdir(exist_ok=True, parents=True)
        paths["output_dir"] = str(output_dir)
    else:
        output_dir = Path(paths["output_dir"])

    # Path to models
    if paths["models_dir"] is None and not train_options["resume"]:
        models_dir = output_dir / "saved_models"
        models_dir.mkdir(exist_ok=True)
        paths["models_dir"] = str(models_dir)
    else:
        models_dir = Path(paths["models_dir"])

    # Log directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    options["paths"] = paths
    with open(output_dir / 'options.json', 'w') as fp:
        json.dump(options, fp, indent=4)

    save_useful_info(output_dir)

    train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        path_processed_data=path_processed_data,
        path_data_definition=path_data_definition,
        train_options=train_options,
        network_options=network_options,
    )

    print("Done!")

import math

import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import interpolate
from scipy.interpolate import griddata

from anim import bvh, quat
from audio.audio_files import read_wavfile, write_wavefile
from audio.spectrograms import extract_mel_spectrogram_for_tts

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_logger = logging.getLogger(__name__)
_logger.propagate = False
warnings.simplefilter("ignore")


# ===============================================
#                   Audio
# ===============================================
def extract_energy(mel_spec):
    energy = np.linalg.norm(mel_spec, axis=0)
    return energy


def preprocess_audio(audio_data, anim_fs, anim_length, params, feature_type):
    if params.normalize_loudness:
        import pyloudnorm as pyln
        meter = pyln.Meter(params.sampling_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio_data)
        # loudness normalize audio to -20 dB LUFS
        audio_data = pyln.normalize.loudness(audio_data, loudness, -20.0)

    resample_method = params.resample_method
    audio_feature = []
    # Extract MEL spectrogram
    mel_spec = extract_mel_spectrogram_for_tts(
        wav_signal=audio_data,
        fs=params.sampling_rate,
        n_fft=params.filter_length,
        step_size=params.hop_length,
        n_mels=params.n_mel_channels,
        mel_fmin=params.mel_fmin,
        mel_fmax=params.mel_fmax,
        min_amplitude=params.min_clipping,
        pre_emphasis=params.pre_emphasis,
        pre_emph_coeff=params.pre_emph_coeff,
        dynamic_range=None,
        real_amplitude=params.real_amplitude,
        centered=params.centered,
        normalize_mel_bins=params.normalize_mel_bins,
        normalize_range=params.normalize_range,
        logger=_logger,
    )[0].T
    mel_spec = 10 ** (mel_spec / 20)
    mel_spec = np.log(mel_spec)

    if "mel_spec" in feature_type:
        mel_spec_interp = interpolate.griddata(
            np.arange(len(mel_spec)),
            mel_spec,
            ((params.sampling_rate / params.hop_length) / anim_fs) * np.arange(anim_length),
            method=resample_method,
        ).astype(np.float32)
        audio_feature.append(mel_spec_interp)

    if "energy" in feature_type:
        energy = extract_energy(np.exp(mel_spec).T)
        f = interpolate.interp1d(np.arange(len(energy)), energy, kind=resample_method, fill_value="extrapolate")
        energy_interp = f(
            ((params.sampling_rate / params.hop_length) / anim_fs) * np.arange(anim_length)
        ).astype(np.float32)
        audio_feature.append(energy_interp[:, np.newaxis])

    audio_feature = np.concatenate(audio_feature, axis=1)

    return audio_feature


# ===============================================
#                   Animation
# ===============================================
def preprocess_animation(anim_data, conf=dict(), animation_path=None, info_df=None, i=0):
    nframes = len(anim_data["rotations"])
    njoints = len(anim_data["parents"])
    dt = anim_data["frametime"]

    lrot = quat.unroll(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))

    lpos = anim_data["positions"]

    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])

    # Find root (Projected hips on the ground)
    root_pos = gpos[:, anim_data["names"].index("Spine2")] * np.array([1, 0, 1])
    # root_pos = signal.savgol_filter(root_pos, 31, 3, axis=0, mode="interp")

    # Root direction
    root_fwd = quat.mul_vec(grot[:, anim_data["names"].index("Hips")], np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]

    # root_fwd = signal.savgol_filter(root_fwd, 61, 3, axis=0, mode="interp")
    # root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]

    # Root rotation
    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )

    # Find look at direction
    gaze_lookat = quat.mul_vec(grot[:, anim_data["names"].index("Head")], np.array([0, 0, 1]))
    gaze_lookat[:, 1] = 0
    gaze_lookat = gaze_lookat / np.sqrt(np.sum(np.square(gaze_lookat), axis=-1))[..., np.newaxis]

    # Find gaze position
    gaze_distance = 100  # Assume other actor is one meter away
    gaze_pos_all = root_pos + gaze_distance * gaze_lookat
    gaze_pos = np.median(gaze_pos_all, axis=0)
    gaze_pos = gaze_pos[np.newaxis].repeat(nframes, axis=0)

    # Visualize Gaze Pos
    if conf.get("visualize_gaze", False):
        import matplotlib.pyplot as plt

        plt.scatter(gaze_pos_all[:, 0], gaze_pos_all[:, 2], s=0.1, marker=".")
        plt.scatter(gaze_pos[0, 0], gaze_pos[0, 2])
        plt.scatter(root_pos[:, 0], root_pos[:, 2], s=0.1, marker=".")
        plt.quiver(root_pos[::60, 0], root_pos[::60, 2], root_fwd[::60, 0], root_fwd[::60, 2])
        plt.gca().set_aspect("equal")
        plt.show()

    # Compute local gaze dir
    gaze_dir = gaze_pos - root_pos
    # gaze_dir = gaze_dir / np.sqrt(np.sum(np.square(gaze_dir), axis=-1))[..., np.newaxis]
    gaze_dir = quat.mul_vec(quat.inv(root_rot), gaze_dir)

    # Make relative to root
    lrot[:, 0] = quat.mul(quat.inv(root_rot), lrot[:, 0])
    lpos[:, 0] = quat.mul_vec(quat.inv(root_rot), lpos[:, 0] - root_pos)

    # Local velocities
    lvel = np.zeros_like(lpos)
    lvel[1:] = (lpos[1:] - lpos[:-1]) / dt
    lvel[0] = lvel[1] - (lvel[3] - lvel[2])

    lvrt = np.zeros_like(lpos)
    lvrt[1:] = quat.to_helical(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) / dt
    lvrt[0] = lvrt[1] - (lvrt[3] - lvrt[2])

    # Root velocities
    root_vrt = np.zeros_like(root_pos)
    root_vrt[1:] = quat.to_helical(quat.abs(quat.mul(root_rot[1:], quat.inv(root_rot[:-1])))) / dt
    root_vrt[0] = root_vrt[1] - (root_vrt[3] - root_vrt[2])
    root_vrt[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vrt[1:])
    root_vrt[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vrt[0])

    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel[0] = root_vel[1] - (root_vel[3] - root_vel[2])
    root_vel[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vel[1:])
    root_vel[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vel[0])

    # Compute character space
    crot, cpos, cvrt, cvel = quat.fk_vel(lrot, lpos, lvrt, lvel, anim_data["parents"])

    # Compute 2-axis transforms
    ltxy = np.zeros(dtype=np.float32, shape=[len(lrot), njoints, 2, 3])
    ltxy[..., 0, :] = quat.mul_vec(lrot, np.array([1.0, 0.0, 0.0]))
    ltxy[..., 1, :] = quat.mul_vec(lrot, np.array([0.0, 1.0, 0.0]))

    ctxy = np.zeros(dtype=np.float32, shape=[len(crot), njoints, 2, 3])
    ctxy[..., 0, :] = quat.mul_vec(crot, np.array([1.0, 0.0, 0.0]))
    ctxy[..., 1, :] = quat.mul_vec(crot, np.array([0.0, 1.0, 0.0]))

    if conf.get("save_normalized_animations", False):
        anim_data["positions"] = lpos
        anim_data["rotations"] = np.degrees(quat.to_euler(lrot, order=anim_data["order"]))

        normalized_animations_path = animation_path / "processed" / "normalized_animations"
        normalized_animations_path.mkdir(exist_ok=True)
        animation_norm_file = str(
            normalized_animations_path / info_df.iloc[i].anim_bvh).replace(
            ".bvh", "_norm.bvh"
        )

        bvh.save(animation_norm_file, anim_data)

        lpos_denorm = lpos.copy()
        lpos_denorm[:, 0] = quat.mul_vec(root_rot, lpos_denorm[:, 0]) + root_pos

        lrot_denorm = lrot.copy()
        lrot_denorm[:, 0] = quat.mul(root_rot, lrot_denorm[:, 0])

        anim_data["positions"] = lpos_denorm
        anim_data["rotations"] = np.degrees(quat.to_euler(lrot_denorm, order=anim_data["order"]))

        animation_denorm_file = str(
            animation_path / "processed" / "normalized_animations" / info_df.iloc[i].anim_bvh
        ).replace(".bvh", "_denorm.bvh")

        bvh.save(animation_denorm_file, anim_data)

    return (
        root_pos,
        root_rot,
        root_vel,
        root_vrt,
        lpos,
        lrot,
        ltxy,
        lvel,
        lvrt,
        cpos,
        crot,
        ctxy,
        cvel,
        cvrt,
        gaze_pos,
        gaze_dir,
    )


# ===============================================
#                   Pipeline
# ===============================================
def data_pipeline(conf):
    """Prepare Audio and Animation data for training

    Args:
        conf: config file

    Returns:
        processed_data, data_definition
    """
    from rich.progress import track
    from rich.console import Console
    from rich.table import Table
    console = Console(record=True)
    console.print("This may take a little bit of time ...")
    len_ratios = conf["len_ratios"]
    base_path = Path(conf["base_path"])
    processed_data_path = base_path / conf["processed_data_path"]
    processed_data_path.mkdir(exist_ok=True)
    info_filename = base_path / conf["info_filename"]
    animation_path = base_path / "original"
    audio_path = base_path / "original"

    with open(str(processed_data_path / "data_pipeline_conf.json"), "w") as f:
        json.dump(conf, f, indent=4)
    conf = DictConfig(conf)

    info_df = pd.read_csv(info_filename)
    num_of_samples = len(info_df)
    audio_desired_fs = conf.audio_conf["sampling_rate"]

    X_audio_features = []

    Y_root_pos = []
    Y_root_rot = []
    Y_root_vrt = []
    Y_root_vel = []

    Y_lpos = []
    Y_lrot = []
    Y_ltxy = []
    Y_lvel = []
    Y_lvrt = []

    Y_gaze_pos = []
    Y_gaze_dir = []

    current_start_frame = 0
    ranges_train = []
    ranges_valid = []
    ranges_train_labels = []
    ranges_valid_labels = []

    for i in track(range(num_of_samples), description="Processing...", complete_style="magenta"):

        animation_file = str(animation_path / info_df.iloc[i].anim_bvh)
        audio_file = audio_path / info_df.iloc[i].audio_filename

        # Load Animation #
        original_anim_data = bvh.load(animation_file)
        anim_fps = int(np.ceil(1 / original_anim_data["frametime"]))
        assert anim_fps == 60

        # Load Audio #
        audio_sr, original_audio_data = read_wavfile(
            audio_file,
            rescale=True,
            desired_fs=audio_desired_fs,
            desired_nb_channels=None,
            out_type="float32",
            logger=_logger,
        )

        # Silence Audio #
        speacker_timing_df = pd.read_csv(audio_file.with_suffix(".csv"))

        # Mark regions that don't need silencing
        mask = np.zeros_like(original_audio_data)
        for ind, row in speacker_timing_df.iterrows():

            if "R" in row["#"]:
                start_time = [int(num) for num in row["Start"].replace(".", ":").rsplit(":")]
                end_time = [int(num) for num in row["End"].replace(".", ":").rsplit(":")]

                start_time = (
                        start_time[0] * 60 * audio_desired_fs
                        + start_time[1] * audio_desired_fs
                        + int(start_time[2] * (audio_desired_fs / 1000))
                )

                end_time = (
                        end_time[0] * 60 * audio_desired_fs
                        + end_time[1] * audio_desired_fs
                        + int(end_time[2] * (audio_desired_fs / 1000))
                )

                mask[start_time:end_time] = 1.0

        # Silence unmarked regions
        original_audio_data = original_audio_data * mask

        # Sync & Trim #
        # Get mark-ups
        audio_start_time = info_df.iloc[i].audio_start_time
        audio_start_time = [int(num) for num in audio_start_time.rsplit(":")]
        anim_start_time = info_df.iloc[i].anim_start_time
        anim_start_time = [int(num) for num in anim_start_time.rsplit(":")]
        acting_start_time = info_df.iloc[i].acting_start_time
        acting_start_time = [int(num) for num in acting_start_time.rsplit(":")]
        acting_end_time = info_df.iloc[i].acting_end_time
        acting_end_time = [int(num) for num in acting_end_time.rsplit(":")]

        # Compute Timings (This is assuming that audio timing is given in 30fps)
        audio_start_time_in_thirds = (
                audio_start_time[0] * 216000
                + audio_start_time[1] * 3600
                + audio_start_time[2] * 60
                + audio_start_time[3] * 2
        )

        anim_start_time_in_thirds = (
                anim_start_time[0] * 216000
                + anim_start_time[1] * 3600
                + anim_start_time[2] * 60
                + anim_start_time[3] * 1
        )

        acting_start_time_in_thirds = (
                acting_start_time[0] * 216000
                + acting_start_time[1] * 3600
                + acting_start_time[2] * 60
                + acting_start_time[3] * 1
        )

        acting_end_time_in_thirds = (
                acting_end_time[0] * 216000
                + acting_end_time[1] * 3600
                + acting_end_time[2] * 60
                + acting_end_time[3] * 1
        )

        acting_start_in_audio_ref = int(
            np.round(
                (acting_start_time_in_thirds - audio_start_time_in_thirds) * (audio_sr / 60)
            )
        )

        acting_end_in_audio_ref = int(
            np.round((acting_end_time_in_thirds - audio_start_time_in_thirds) * (audio_sr / 60))
        )

        acting_start_in_anim_ref = int(
            np.round(
                (acting_start_time_in_thirds - anim_start_time_in_thirds) * (anim_fps / 60)
            )
        )

        acting_end_in_anim_ref = int(
            np.round((acting_end_time_in_thirds - anim_start_time_in_thirds) * (anim_fps / 60))
        )

        if (
                acting_start_in_audio_ref < 0
                or acting_start_in_anim_ref < 0
                or acting_end_in_audio_ref < 0
                or acting_end_in_anim_ref < 0
        ):
            raise ValueError("The timings are incorrect!")

        # Trim to equal length
        original_audio_data = original_audio_data[acting_start_in_audio_ref:acting_end_in_audio_ref]

        original_anim_data["rotations"] = original_anim_data["rotations"][
                                          acting_start_in_anim_ref:acting_end_in_anim_ref
                                          ]

        original_anim_data["positions"] = original_anim_data["positions"][
                                          acting_start_in_anim_ref:acting_end_in_anim_ref
                                          ]
        for len_ratio in len_ratios:
            anim_data = original_anim_data.copy()
            audio_data = original_audio_data.copy()
            if len_ratio != 1.0:
                n_anim_frames = len(original_anim_data["rotations"])
                nbones = anim_data["positions"].shape[1]
                original_times = np.linspace(0, n_anim_frames - 1, n_anim_frames)
                sample_times = np.linspace(0, n_anim_frames - 1, int(len_ratio * (n_anim_frames)))
                anim_data["positions"] = griddata(original_times, anim_data["positions"].reshape([n_anim_frames, -1]),
                                                  sample_times, method='cubic').reshape([len(sample_times), nbones, 3])

                rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
                rotations = griddata(original_times, rotations.reshape([n_anim_frames, -1]), sample_times,
                                     method='cubic').reshape([len(sample_times), nbones, 4])
                rotations = quat.normalize(rotations)
                anim_data["rotations"] = np.degrees(quat.to_euler(rotations, order=anim_data["order"]))

                n_audio_frames = len(audio_data)
                original_times = np.linspace(0, n_audio_frames - 1, n_audio_frames)
                sample_times = np.linspace(0, n_audio_frames - 1, int(len_ratio * (n_audio_frames)))
                audio_data = griddata(original_times, audio_data, sample_times, method='cubic')
                # assert len(audio_data) / audio_sr == len(anim_data["rotations"]) / anim_fps

            # Saving Trimmed Files
            folder = "valid" if info_df.iloc[i].validation else "train"
            trimmed_filename = info_df.iloc[i].anim_bvh.split(".")[0]
            trimmed_filename = trimmed_filename + "_x_" + str(len_ratio).replace(".", "_")

            if conf["save_trimmed_audio"]:
                target_path = processed_data_path / "trimmed" / folder
                target_path.mkdir(exist_ok=True, parents=True)
                write_wavefile(target_path / (trimmed_filename + ".wav"), audio_data, audio_sr)

            if conf["save_trimmed_animation"]:
                target_path = processed_data_path / "trimmed" / folder
                target_path.mkdir(exist_ok=True, parents=True)

                # Centering the character. Comment if you want the original global position and orientation
                output = anim_data.copy()
                lrot = quat.from_euler(np.radians(output["rotations"]), output["order"])
                offset_pos = output["positions"][0:1, 0:1].copy() * np.array([1, 0, 1])
                offset_rot = lrot[0:1, 0:1].copy() * np.array([1, 0, 1, 0])

                root_pos = quat.mul_vec(quat.inv(offset_rot), output["positions"][:, 0:1] - offset_pos)
                output["positions"][:, 0:1] = quat.mul_vec(quat.inv(offset_rot),
                                                           output["positions"][:, 0:1] - offset_pos)
                output["rotations"][:, 0:1] = np.degrees(
                    quat.to_euler(quat.mul(quat.inv(offset_rot), lrot[:, 0:1]), order=output["order"]))

                bvh.save(target_path / (trimmed_filename + ".bvh"), anim_data)

            # Extracting Audio Features #
            audio_features = preprocess_audio(
                audio_data,
                anim_fps,
                len(anim_data["rotations"]),
                conf.audio_conf,
                feature_type=conf.audio_feature_type,
            )

            # Check if the lengths are correct and no NaNs
            assert len(audio_features) == len(anim_data["rotations"])
            assert not np.any(np.isnan(audio_features))

            if conf["visualize_spectrogram"]:
                import matplotlib.pyplot as plt

                plt.imshow(audio_features.T, interpolation="nearest")
                plt.show()

            # Extracting Animation Features
            nframes = len(anim_data["rotations"])
            dt = anim_data["frametime"]
            (
                root_pos,
                root_rot,
                root_vel,
                root_vrt,
                lpos,
                lrot,
                ltxy,
                lvel,
                lvrt,
                cpos,
                crot,
                ctxy,
                cvel,
                cvrt,
                gaze_pos,
                gaze_dir,
            ) = preprocess_animation(anim_data, conf, animation_path, info_df, i)

            # Appending Data
            X_audio_features.append(audio_features)

            Y_root_pos.append(root_pos)
            Y_root_rot.append(root_rot)
            Y_root_vel.append(root_vel)
            Y_root_vrt.append(root_vrt)
            Y_lpos.append(lpos)
            Y_lrot.append(lrot)
            Y_ltxy.append(ltxy)
            Y_lvel.append(lvel)
            Y_lvrt.append(lvrt)

            Y_gaze_pos.append(gaze_pos)
            Y_gaze_dir.append(gaze_dir)

            # Append to Ranges
            current_end_frame = nframes + current_start_frame

            if info_df.iloc[i].validation:
                ranges_valid.append([current_start_frame, current_end_frame])
                ranges_valid_labels.append(info_df.iloc[i].style)
            else:
                ranges_train.append([current_start_frame, current_end_frame])
                ranges_train_labels.append(info_df.iloc[i].style)

            current_start_frame = current_end_frame

    # Processing Labels
    ranges_train = np.array(ranges_train, dtype=np.int32)
    ranges_valid = np.array(ranges_valid, dtype=np.int32)

    label_names = list(set(ranges_train_labels + ranges_valid_labels))

    ranges_train_labels = np.array(
        [label_names.index(label) for label in ranges_train_labels], dtype=np.int32
    )
    ranges_valid_labels = np.array(
        [label_names.index(label) for label in ranges_valid_labels], dtype=np.int32
    )

    # Concatenating Data
    X_audio_features = np.concatenate(X_audio_features, axis=0).astype(np.float32)

    Y_root_pos = np.concatenate(Y_root_pos, axis=0).astype(np.float32)
    Y_root_rot = np.concatenate(Y_root_rot, axis=0).astype(np.float32)
    Y_root_vel = np.concatenate(Y_root_vel, axis=0).astype(np.float32)
    Y_root_vrt = np.concatenate(Y_root_vrt, axis=0).astype(np.float32)

    Y_lpos = np.concatenate(Y_lpos, axis=0).astype(np.float32)
    Y_lrot = np.concatenate(Y_lrot, axis=0).astype(np.float32)
    Y_ltxy = np.concatenate(Y_ltxy, axis=0).astype(np.float32)
    Y_lvel = np.concatenate(Y_lvel, axis=0).astype(np.float32)
    Y_lvrt = np.concatenate(Y_lvrt, axis=0).astype(np.float32)

    Y_gaze_pos = np.concatenate(Y_gaze_pos, axis=0).astype(np.float32)
    Y_gaze_dir = np.concatenate(Y_gaze_dir, axis=0).astype(np.float32)

    # Compute Means & Stds
    # Filter out start and end frames
    ranges_mask = np.zeros(len(X_audio_features), dtype=bool)
    for s, e in ranges_train:
        ranges_mask[s + 2: e - 2] = True

    # Compute Means
    Y_root_vel_mean = Y_root_vel[ranges_mask].mean(axis=0)
    Y_root_vrt_mean = Y_root_vrt[ranges_mask].mean(axis=0)

    Y_lpos_mean = Y_lpos[ranges_mask].mean(axis=0)
    Y_ltxy_mean = Y_ltxy[ranges_mask].mean(axis=0)
    Y_lvel_mean = Y_lvel[ranges_mask].mean(axis=0)
    Y_lvrt_mean = Y_lvrt[ranges_mask].mean(axis=0)

    Y_gaze_dir_mean = Y_gaze_dir[ranges_mask].mean(axis=0)

    audio_input_mean = X_audio_features[ranges_mask].mean(axis=0)

    anim_input_mean = np.hstack(
        [
            Y_root_vel_mean.ravel(),
            Y_root_vrt_mean.ravel(),
            Y_lpos_mean.ravel(),
            Y_ltxy_mean.ravel(),
            Y_lvel_mean.ravel(),
            Y_lvrt_mean.ravel(),
            Y_gaze_dir_mean.ravel(),
        ]
    )

    # Compute Stds
    Y_root_vel_std = Y_root_vel[ranges_mask].std() + 1e-10
    Y_root_vrt_std = Y_root_vrt[ranges_mask].std() + 1e-10

    Y_lpos_std = Y_lpos[ranges_mask].std() + 1e-10
    Y_ltxy_std = Y_ltxy[ranges_mask].std() + 1e-10
    Y_lvel_std = Y_lvel[ranges_mask].std() + 1e-10
    Y_lvrt_std = Y_lvrt[ranges_mask].std() + 1e-10

    Y_gaze_dir_std = Y_gaze_dir[ranges_mask].std() + 1e-10

    audio_input_std = X_audio_features[ranges_mask].std() + 1e-10

    anim_input_std = np.hstack(
        [
            Y_root_vel_std.repeat(len(Y_root_vel_mean.ravel())),
            Y_root_vrt_std.repeat(len(Y_root_vrt_mean.ravel())),
            Y_lpos_std.repeat(len(Y_lpos_mean.ravel())),
            Y_ltxy_std.repeat(len(Y_ltxy_mean.ravel())),
            Y_lvel_std.repeat(len(Y_lvel_mean.ravel())),
            Y_lvrt_std.repeat(len(Y_lvrt_mean.ravel())),
            Y_gaze_dir_std.repeat(len(Y_gaze_dir_mean.ravel())),
        ]
    )

    # Compute Output Means
    anim_output_mean = np.hstack(
        [
            Y_root_vel_mean.ravel(),
            Y_root_vrt_mean.ravel(),
            Y_lpos_mean.ravel(),
            Y_ltxy_mean.ravel(),
            Y_lvel_mean.ravel(),
            Y_lvrt_mean.ravel(),
        ]
    )

    # Compute Output Stds
    Y_root_vel_out_std = Y_root_vel[ranges_mask].std(axis=0)
    Y_root_vrt_out_std = Y_root_vrt[ranges_mask].std(axis=0)

    Y_lpos_out_std = Y_lpos[ranges_mask].std(axis=0)
    Y_ltxy_out_std = Y_ltxy[ranges_mask].std(axis=0)
    Y_lvel_out_std = Y_lvel[ranges_mask].std(axis=0)
    Y_lvrt_out_std = Y_lvrt[ranges_mask].std(axis=0)

    anim_output_std = np.hstack(
        [
            Y_root_vel_out_std.ravel(),
            Y_root_vrt_out_std.ravel(),
            Y_lpos_out_std.ravel(),
            Y_ltxy_out_std.ravel(),
            Y_lvel_out_std.ravel(),
            Y_lvrt_out_std.ravel(),
        ]
    )

    processed_data = dict(
        X_audio_features=X_audio_features,
        Y_root_pos=Y_root_pos,
        Y_root_rot=Y_root_rot,
        Y_root_vel=Y_root_vel,
        Y_root_vrt=Y_root_vrt,
        Y_lpos=Y_lpos,
        Y_ltxy=Y_ltxy,
        Y_lvel=Y_lvel,
        Y_lvrt=Y_lvrt,
        Y_gaze_pos=Y_gaze_pos,
        ranges_train=ranges_train,
        ranges_valid=ranges_valid,
        ranges_train_labels=ranges_train_labels,
        ranges_valid_labels=ranges_valid_labels,
        audio_input_mean=audio_input_mean,
        audio_input_std=audio_input_std,
        anim_input_mean=anim_input_mean,
        anim_input_std=anim_input_std,
        anim_output_mean=anim_output_mean,
        anim_output_std=anim_output_std,
    )

    stats = dict(
        ranges_train=ranges_train,
        ranges_valid=ranges_valid,
        ranges_train_labels=ranges_train_labels,
        ranges_valid_labels=ranges_valid_labels,
        audio_input_mean=audio_input_mean,
        audio_input_std=audio_input_std,
        anim_input_mean=anim_input_mean,
        anim_input_std=anim_input_std,
        anim_output_mean=anim_output_mean,
        anim_output_std=anim_output_std,
    )

    data_definition = dict(
        dt=dt,
        label_names=label_names,
        parents=anim_data["parents"].tolist(),
        bone_names=anim_data["names"],
    )

    # Save Data
    if conf["save_final_data"]:
        np.savez(processed_data_path / "processed_data.npz", **processed_data)

        np.savez(processed_data_path / "stats.npz", **stats)

        with open(str(processed_data_path / "data_definition.json"), "w") as f:
            json.dump(data_definition, f, indent=4)

    # Data Stats:
    nlabels = len(label_names)
    df = pd.DataFrame()
    df["Dataset"] = ["Train", "Validation", "Total"]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    table = Table(title="Data Info", show_lines=True, row_styles=["magenta"])
    table.add_column("Dataset")
    data_len = 0
    for i in range(nlabels):
        ind_mask = ranges_train_labels == i
        ranges = ranges_train[ind_mask]
        num_train_frames = (
                np.sum(ranges[:, 1] - ranges[:, 0]) / 2
        )  # It is divided by two as we have mirrored versions too
        ind_mask = ranges_valid_labels == i
        ranges = ranges_valid[ind_mask]
        num_valid_frames = np.sum(ranges[:, 1] - ranges[:, 0]) / 2
        total = num_train_frames + num_valid_frames
        df[label_names[i]] = [
            f"{num_train_frames} frames - {num_train_frames / 60:.1f} secs",
            f"{num_valid_frames} frames - {num_valid_frames / 60:.1f} secs",
            f"{total} frames - {total / 60:.1f} secs",
        ]
        table.add_column(label_names[i])
        data_len += total

    for i in range(3):
        table.add_row(*list(df.iloc[i]))
    console.print(table)
    console.print(f"Total length of dataset is {data_len} frames - {data_len / 60:.1f} seconds")
    console_print_file = processed_data_path / "data_info.html"
    console.print(dict(conf))
    console.save_html(str(console_print_file))

    return processed_data, data_definition


if __name__ == "__main__":
    config_file = "../configs/data_pipeline_conf_v1.json"
    with open(config_file, "r") as f:
        conf = json.load(f)

    data_pipeline(conf)

import datetime
import os
import shutil


def save_useful_info(dest_path) -> None:
    dst = os.path.join(dest_path, "code")
    if not os.path.exists(dst):
        shutil.copytree(
            os.getcwd(), dst,
        )


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def split_by_ratio(length, ratio):
    assert sum(ratio) == 1.0
    se = [0, 0]
    split = []
    for r in ratio:
        l = r * length
        s = int(se[-1])
        e = int(se[-1] + l)
        se = [s, e]
        split.append(se)
    split[-1][-1] = length
    return split


def percent_bar(ratio=1.0, width=30, empty=' ', done='#', parts=' -=>'):
    if ratio == 1.0:
        return done * width
    else:
        return (
                (done * width)[:int((100 * ratio) // (100 / width))] +
                (parts)[int(len(parts) * (((100 * ratio) / (100 / width)) % 1.0))] +
                (empty * width)[:max(width - int((100 * ratio) // (100 / width)) - 1, 0)])


def progress(ei, ii, bi, train_err, iter_num, start_time):
    percent = (float(bi) + 1) / (iter_num)
    curr_time = datetime.datetime.now()
    eta_time = start_time + (1.0 / (percent + 1e-10)) * (curr_time - start_time)

    return ("| %5i | %6i | [%s] %6.2f%% | % 8.4f | %s |" %
            (ei, ii, percent_bar(percent), 100 * percent, train_err, str(eta_time)[11:19]))

import numpy as np

from . import mat


def to_translation(x):
    return x[..., :3, 3] / x[..., 3, 3][..., np.newaxis]


def to_rotation(x):
    return x[..., :3, :3]


def to_rotation_translation(x):
    return to_rotation(x), to_translation(x)


def log(x, eps=1e-10):
    angle, axis = to_angle_axis(x, eps=eps)
    return (angle / 2.0)[..., np.newaxis] * axis


def exp(x, eps=1e-10):
    halfangle = np.sqrt(np.sum(x ** 2.0, axis=-1))
    axis = x[..., :3] / (halfangle[..., np.newaxis] + eps)
    return from_angle_axis(2.0 * halfangle, axis)


def to_angle_axis(x, eps=1e-10):
    angle = np.arccos(np.clip((x[..., 0, 0] + x[..., 1, 1] + x[..., 2, 2] - 1.0) / 2.0, 0.0, 1.0))
    axis = np.concatenate([
        (x[..., 2, 1] - x[..., 1, 2])[..., np.newaxis],
        (x[..., 0, 2] - x[..., 2, 0])[..., np.newaxis],
        (x[..., 1, 0] - x[..., 0, 1])[..., np.newaxis]
    ], axis=-1) / ((2.0 * np.sin(angle))[..., np.newaxis] + eps)

    return angle, axis


def from_rotation_translation(rot, pos):
    x = np.concatenate([rot, pos[..., np.newaxis]], axis=-1)
    x = np.concatenate([x, np.ones_like(x)[..., :1, :] * np.array([0, 0, 0, 1], dtype=np.float32)], axis=-2)
    return x


def from_angle_axis(angle, axis):
    angle = angle[..., np.newaxis]
    a0, a1, a2 = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)

    r0 = np.concatenate([c + a0 * a0 * t, a0 * a1 * t - a2 * s, a0 * a2 * t + a1 * s], axis=-1)
    r1 = np.concatenate([a0 * a1 * t + a2 * s, c + a1 * a1 * t, a1 * a2 * t - a0 * s], axis=-1)
    r2 = np.concatenate([a0 * a2 * t - a1 * s, a1 * a2 * t + a0 * s, c + a2 * a2 * t], axis=-1)

    return np.concatenate([r0[..., np.newaxis, :], r1[..., np.newaxis, :], r2[..., np.newaxis, :]], axis=-2)


def from_euler(e, order='zyx'):
    c, s = np.cos(e), np.sin(e)
    c0, c1, c2 = c[..., 0:1], c[..., 1:2], c[..., 2:3]
    s0, s1, s2 = s[..., 0:1], s[..., 1:2], s[..., 2:3]

    if order == 'xzy':
        r0 = np.concatenate([c1 * c2, -s1, c1 * s2], axis=-1)
        r1 = np.concatenate([s0 * s2 + c0 * c2 * s1, c0 * c1, c0 * s1 * s2 - c2 * s0], axis=-1)
        r2 = np.concatenate([c2 * s0 * s1 - c0 * s2, c1 * s0, c0 * c2 + s0 * s1 * s2], axis=-1)
    elif order == 'xyz':
        r0 = np.concatenate([c1 * c2, -c1 * s2, s1], axis=-1)
        r1 = np.concatenate([c0 * s2 + c2 * s0 * s1, c0 * c2 - s0 * s1 * s2, -c1 * s0], axis=-1)
        r2 = np.concatenate([s0 * s2 - c0 * c2 * s1, c2 * s0 + c0 * s1 * s2, c0 * c1], axis=-1)
    elif order == 'yxz':
        r0 = np.concatenate([c0 * c2 + s0 * s1 * s2, c2 * s0 * s1 - c0 * s2, c1 * s0], axis=-1)
        r1 = np.concatenate([c1 * s2, c1 * c2, -s1], axis=-1)
        r2 = np.concatenate([c0 * s1 * s2 - c2 * s0, c0 * c2 * s1 + s0 * s2, c0 * c1], axis=-1)
    elif order == 'yzx':
        r0 = np.concatenate([c0 * c1, s0 * s2 - c0 * c2 * s1, c2 * s0 + c0 * s1 * s2], axis=-1)
        r1 = np.concatenate([s1, c1 * c2, -c1 * s2], axis=-1)
        r2 = np.concatenate([-c1 * s0, c0 * s2 + c2 * s0 * s1, c0 * c2 - s0 * s1 * s2], axis=-1)
    elif order == 'zyx':
        r0 = np.concatenate([c0 * c1, c0 * s1 * s2 - c2 * s0, s0 * s2 + c0 * c2 * s1], axis=-1)
        r1 = np.concatenate([c1 * s0, c0 * c2 + s0 * s1 * s2, c2 * s0 * s1 - c0 * s2], axis=-1)
        r2 = np.concatenate([-s1, c1 * s2, c1 * c2], axis=-1)
    elif order == 'zxy':
        r0 = np.concatenate([c0 * c2 - s0 * s1 * s2, -c1 * s0, c0 * s2 + c2 * s0 * s1], axis=-1)
        r1 = np.concatenate([c2 * s0 + c0 * s1 * s2, c0 * c1, s0 * s2 - c0 * c2 * s1], axis=-1)
        r2 = np.concatenate([-c1 * s2, s1, c1 * c2], axis=-1)
    else:
        raise Exception('Unknown ordering: %s' % order)

    return np.concatenate([r0[..., np.newaxis, :], r1[..., np.newaxis, :], r2[..., np.newaxis, :]], axis=-2)


def from_basis(x, y, z):
    return np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)


def orthogonalize(x, method='svd', eps=0.0):
    def cross(a, b):
        return np.concatenate([
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ], axis=-1)

    if method == 'cross':
        r0, r1 = x[..., 0], x[..., 1]
        r2 = cross(r0, r1)
        r0 = r0 / (np.sqrt(np.sum(r0 * r0, axis=-1))[..., np.newaxis] + eps)
        r2 = r2 / (np.sqrt(np.sum(r2 * r2, axis=-1))[..., np.newaxis] + eps)
        r1 = cross(r2, r0)
        return from_basis(r0, r1, r2)
    elif method == 'svd':
        U, _, V = mat.svd(x)
        return mat.mul(U, V)
    else:
        raise ValueError('Unknown method \'%s\'' % method)


def orthogonalize_from_xy(xy):
    xaxis = xy[..., 0:1, :]
    zaxis = np.cross(xaxis, xy[..., 1:2, :])
    yaxis = np.cross(zaxis, xaxis)

    output = np.concatenate([
        xaxis / np.sqrt(np.sum(xaxis * xaxis, axis=-1))[..., np.newaxis],
        yaxis / np.sqrt(np.sum(yaxis * yaxis, axis=-1))[..., np.newaxis],
        zaxis / np.sqrt(np.sum(zaxis * zaxis, axis=-1))[..., np.newaxis]
    ], axis=-2)

    return mat.transpose(output)


def orthogonalize_iterative(ts, R=None, iterations=4, eps=1e-5):
    if R is None:
        R = np.zeros_like(ts)
        R[..., :, :] = np.eye(3)

    for _ in range(iterations):
        omega = ((
                         np.cross(R[..., :, 0], ts[..., :, 0]) +
                         np.cross(R[..., :, 1], ts[..., :, 1]) +
                         np.cross(R[..., :, 2], ts[..., :, 2])
                 ) / (abs(
            np.sum(R[..., :, 0] * ts[..., :, 0], axis=-1)[..., np.newaxis] +
            np.sum(R[..., :, 1] * ts[..., :, 1], axis=-1)[..., np.newaxis] +
            np.sum(R[..., :, 2] * ts[..., :, 2], axis=-1)[..., np.newaxis]
        ) + eps))

        w = np.sqrt(np.sum(np.square(omega), axis=-1))

        R = mat.mul(from_angle_axis(w, omega / (w[..., np.newaxis] + eps)), R)

    return R

import torch


def xform_transpose(xform):
    s = list(range(len(xform.shape)))
    s[-1], s[-2] = s[-2], s[-1]
    return xform.permute(*s)


def xform_fk_vel(lxform, lpos, lvrt, lvel, parents):
    gr, gp, gt, gv = [lxform[..., :1, :, :]], [lpos[..., :1, :]], [lvrt[..., :1, :]], [lvel[..., :1, :]]
    for i in range(1, len(parents)):
        p = parents[i]
        gp.append(gp[p] + torch.matmul(gr[p], lpos[..., i:i + 1, :][..., None])[..., 0])
        gr.append(torch.matmul(gr[p], lxform[..., i:i + 1, :, :]))
        gt.append(gt[p] + torch.matmul(gr[p], lvrt[..., i:i + 1, :][..., None])[..., 0])
        gv.append(gv[p] + torch.matmul(gr[p], lvel[..., i:i + 1, :][..., None])[..., 0] +
                  torch.cross(gt[p], torch.matmul(gr[p], lpos[..., i:i + 1, :][..., None])[..., 0], dim=-1))

    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2), torch.cat(gt, dim=-2), torch.cat(gv, dim=-2)


def xform_orthogonalize_from_xy(xy, eps=1e-10):
    xaxis = xy[..., 0:1, :]
    zaxis = torch.cross(xaxis, xy[..., 1:2, :])
    yaxis = torch.cross(zaxis, xaxis)

    output = torch.cat([
        xaxis / (torch.norm(xaxis, 2, dim=-1)[..., None] + eps),
        yaxis / (torch.norm(yaxis, 2, dim=-1)[..., None] + eps),
        zaxis / (torch.norm(zaxis, 2, dim=-1)[..., None] + eps)
    ], dim=-2)

    return xform_transpose(output)

import numpy as np

def mul(x, y):
    return np.matmul(x, y)

def mul_vec(x, y):
    return np.matmul(x, y[...,np.newaxis])[...,0]
    
def transpose(x):
    return x.transpose(list(range(x.ndim-2))+[x.ndim-1, x.ndim-2])
    
# TODO: Should work for multiple matrices at once.
def psolve(x, y, eps=1e-5):
    if x.shape[0] < x.shape[1]:
        return np.linalg.solve(x.T.dot(x) + (eps*eps) * np.eye(x.shape[1]), x.T.dot(y)).T
    elif x.shape[0] > x.shape[1]:
        return x.T.dot(np.linalg.solve(x.dot(x.T) + (eps*eps) * np.eye(x.shape[0]), y)).T
    else:
        return np.linalg.solve(x + (eps*eps) * np.eye(x.shape[0]), y).T
import re
import numpy as np

def load(filename, start=None, end=None, order=None):
    
    channelmap = {
        'Xrotation' : 'x',
        'Yrotation' : 'y',
        'Zrotation' : 'z'   
    }
    
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False
    state = 'definition'
    
    names   = []
    offsets = np.empty(shape=[0, 3], dtype=np.float32)
    parents = np.empty(shape=[0],    dtype=np.int32)
    
    for line in f:
        
        if state == 'definition':
        
            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]], dtype=np.float32), axis=0)
                parents = np.append(parents, np.array([active], dtype=np.int32))
                active  = parents.shape[0]-1
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site: end_site = False
                else: active = parents[active]
                continue
            
            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array(list(map(float, offmatch.groups())))
                continue
               
            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2+channelis:2+channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]], dtype=np.float32), axis=0)
                parents = np.append(parents, np.array([active], dtype=np.int32))
                active  = (parents.shape[0]-1)
                continue
            
            if "End Site" in line:
                end_site = True
                continue
                  
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                if start and end:
                    fnum = (end - start)-1
                else:
                    fnum = int(fmatch.group(1))
                jnum = parents.shape[0]
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros([fnum, jnum, 3], dtype=np.float32)
                continue
            
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                state = 'body'
                continue
            
        elif state == 'body':
            
            if (start and end) and (i < start or i >= end-1):
                i += 1
                continue
            
            dmatch = line.strip().split()
            if dmatch:
                
                fi = i - start if start else i
                data_block = np.asarray(tuple(map(float, dmatch)))
                N = parents.shape[0]
                if   channels == 3:
                    positions[fi,0] = data_block[0:3]
                    rotations[fi,:] = data_block[3: ].reshape([N, 3])
                elif channels == 6:
                    data_block = data_block.reshape([N, 6])
                    positions[fi,:] = data_block[:,0:3]
                    rotations[fi,:] = data_block[:,3:6]
                elif channels == 9:
                    positions[fi,0] = data_block[0:3]
                    data_block = data_block[3:].reshape([N-1, 9])
                    rotations[fi,1:] = data_block[:,3:6]
                    positions[fi,1:] = positions[fi,1:] + data_block[:,0:3] * data_block[:,6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1
        
        else:
        
            raise Exception()
        
    f.close()
    
    return {
        'rotations': rotations,
        'positions': positions,
        'offsets': offsets,
        'parents': parents,
        'names': names,
        'order': order,
        'frametime': frametime
    }

def save(filename, data, translations=False):

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }

    rots, poss, offsets, parents = [
        data['rotations'],
        data['positions'],
        data['offsets'],
        data['parents']]
    
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0/60.0)
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % ((t,) + tuple(offsets[0])))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], 
                channelmap_inv[order[1]], 
                channelmap_inv[order[2]]))
        jseq = [0]       
        for i in range(len(parents)):
            if parents[i] == 0:
                t, jseq = save_joint(f, offsets, order, parents, names, t, i, jseq, translations=translations)

        t = t[:-1]
        f.write("%s}\n" % t)
        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots))
        f.write("Frame Time: %f\n" % frametime)
        
        for i in range(rots.shape[0]):
            for j in jseq:
                
                if translations or j == 0:
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                
                else:   
                    f.write("%f %f %f " % (
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))

            f.write("\n")
    
def save_joint(f, offsets, order, parents, names, t, i, jseq, translations=False):

    jseq.append(i)

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % ((t,) + tuple(offsets[i])))
    
    if translations:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(len(parents)):
        if parents[j] == i:
            t, jseq = save_joint(f, offsets, order, parents, names, t, j, jseq, translations=translations)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t, jseq


import torch
import math
from typing import Tuple

@torch.jit.script
def quat_mul(x, y):
    
    x0, x1, x2, x3 = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    y0, y1, y2, y3 = y[...,0:1], y[...,1:2], y[...,2:3], y[...,3:4]

    return torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,      
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,   
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,    
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)
    
@torch.jit.script
def quat_mul_vec(x, y):
    t = 2.0 * torch.cross(x[...,1:], y, dim=-1)
    return y + x[...,0:1] * t + torch.cross(x[...,1:], t, dim=-1)

@torch.jit.script
def quat_inv(x):
    return torch.tensor([1,-1,-1,-1], dtype=torch.float32, device=x.device) * x

@torch.jit.script
def quat_inv_mul(x, y):
    return quat_mul(quat_inv(x), y)
    
@torch.jit.script
def quat_inv_mul_vec(x, y):
    return quat_mul_vec(quat_inv(x), y)

@torch.jit.script
def quat_abs(x):
    return torch.where(x[...,0:1] > 0.0, x, -x)

@torch.jit.script
def quat_diff(x, y, world : bool = True):
    diff = torch.sum(x * y, dim=-1, keepdim=True)
    flip = torch.where(diff > 0.0, x, -x)
    return quat_mul(flip, quat_inv(y)) if world else quat_mul(quat_inv(y), flip)

@torch.jit.script
def quat_diff_linear(x, y):
    d = quat_abs(quat_mul(x, quat_inv(y)))
    return 2.0 * d[...,1:]
    
@torch.jit.script
def quat_normalize(x, eps : float = 1e-5):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

@torch.jit.script
def quat_to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)[...,None,:],
    ], dim=-2)
    
@torch.jit.script
def quat_to_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz)], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx        ], dim=-1)[...,None,:],
    ], dim=-2)
    
@torch.jit.script
def quat_log(x, eps : float = 1e-5):
    length = torch.norm(x[...,1:], dim=-1, keepdim=True)
    return torch.where(
        length < eps, 
        x[...,1:], 
        (torch.atan2(length, x[...,0:1]) / length) * x[...,1:])
    
@torch.jit.script
def quat_exp(x, eps : float = 1e-5):
    halfangle = torch.norm(x, dim=-1, keepdim=True)
    return torch.where(halfangle < eps, quat_normalize(
        torch.cat([torch.ones_like(halfangle), x], dim=-1)),
        torch.cat([torch.cos(halfangle),       x * torch.sinc(halfangle / math.pi)], dim=-1))
    
@torch.jit.script
def quat_to_helical(x, eps : float = 1e-5):
    return 2.0 * quat_log(x, eps)
    
@torch.jit.script
def quat_from_helical(x, eps : float = 1e-5):
    return quat_exp(x / 2.0, eps)
    
@torch.jit.script
def quat_from_helical_approx(v):
    return quat_normalize(torch.cat([
        torch.ones_like(v[...,:1]), v / 2.0], dim=-1))
        
@torch.jit.script
def quat_to_helical_approx(v):
    return 2.0 * v[...,:1]
    
@torch.jit.script
def quat_from_angle_axis(angle, axis):
    c = torch.cos(angle / 2.0)[...,None]
    s = torch.sin(angle / 2.0)[...,None]
    return torch.cat([c, s * axis], dim=-1)
    
@torch.jit.script
def quat_to_angle_axis(x, eps : float = 1e-5):
    length = torch.norm(x[...,1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, x[...,0])
    return angle, x[...,1:] / (length + eps)
    
@torch.jit.script
def quat_ik_rot(grot, parents):
    lr = [grot[...,:1,:]]
    for i in range(1, len(parents)):
        p = parents[i]
        lr.append(quat_mul(quat_inv(grot[...,p:p+1,:]), grot[...,i:i+1,:]))
    return  torch.cat(lr, dim=-2)
    
@torch.jit.script
def quat_fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

@torch.jit.script
def quat_fk_vel(lrot, lpos, lvrt, lvel, parents):
    
    gp, gr, gt, gv = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvrt[...,:1,:]], [lvel[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gt.append(gt[parents[i]] + quat_mul_vec(gr[parents[i]], lvrt[...,i:i+1,:]))
        gv.append(gv[parents[i]] + quat_mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            torch.cross(gt[parents[i]], quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]), dim=-1))
        
    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2), torch.cat(gt, dim=-2), torch.cat(gv, dim=-2)

@torch.jit.script
def quat_fk_root_position(gpos, grot, lpos, parents):
    gp = [gpos[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(grot[...,parents[i]:parents[i]+1,:], lpos[...,i:i+1,:]) + gp[parents[i]])
    return torch.cat(gp, dim=-2)
    
@torch.jit.script
def quat_character_to_local(grot, parents):
    lr = [grot[...,:1,:]]
    for i in range(1, len(parents)):
        lr.append(quat_mul( quat_inv(grot[...,parents[i]:parents[i]+1,:]), grot[...,i:i+1,:]))
    return torch.cat(lr, dim=-2)



import numpy as np


def eye(shape=[]):
    if shape == []:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    else:
        return np.array([1, 0, 0, 0], dtype=np.float32) * np.ones(
            np.concatenate([shape, [4]], axis=0), dtype=np.float32)


def eye_like(x):
    return np.array([1, 0, 0, 0], dtype=np.float32) * np.ones_like(
        x[..., np.newaxis].repeat(4, axis=-1))


def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)


def _fast_cross(a, b):
    o = np.empty(np.broadcast(a, b).shape)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o


def mul_vec(x, y):
    t = 2.0 * _fast_cross(x[..., 1:], y)
    return y + x[..., 0][..., np.newaxis] * t + _fast_cross(x[..., 1:], t)


def mul_scalar(x, y):
    return slerp(eye_like(x[..., 0]), x, y)


def inv(x):
    return np.array([1, -1, -1, -1], dtype=np.float32) * x


def abs(x):
    return np.where((np.sum(x * np.array([1, 0, 0, 0], dtype=np.float32), axis=-1) > 0.0)[..., np.newaxis], x, -x)


def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))[..., np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[..., 0:1]) / length)
    return halfangle * x[..., 1:]


def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[..., np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)


def to_helical(x, eps=1e-5):
    return 2.0 * log(x, eps)


def from_helical(x, eps=1e-5):
    return exp(x / 2.0, eps)


def to_angle_axis(x, eps=1e-10):
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))
    angle = 2.0 * np.arctan2(length, x[..., 0])
    return angle, x[..., 1:] / (length + eps)


def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    return np.concatenate([c, s * axis], axis=-1)


def diff(x, y, world=True):
    diff = np.sum(x * y, axis=-1)[..., np.newaxis]
    flip = np.where(diff > 0.0, x, -x)
    return mul(flip, inv(y)) if world else mul(inv(y), flip)


def normalize(x, eps=0.0):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + eps)


def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        _fast_cross(x, y)], axis=-1)


def slerp(x, y, a, eps=1e-10):
    l = np.sum(x * y, axis=-1)
    o = np.arccos(np.clip(l, -1.0, 1.0))
    a0 = np.sin((1.0 - a) * o) / (np.sin(o) + eps)
    a1 = np.sin((a) * o) / (np.sin(o) + eps)
    return a0[..., np.newaxis] * x + a1[..., np.newaxis] * y


def to_euler(x, order='zyx'):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    if order == 'zyx':
        return np.concatenate([
            np.arctan2(2.0 * (x0 * x3 + x1 * x2), 1.0 - 2.0 * (x2 * x2 + x3 * x3)),
            np.arcsin(np.clip(2.0 * (x0 * x2 - x3 * x1), -1.0, 1.0)),
            np.arctan2(2.0 * (x0 * x1 + x2 * x3), 1.0 - 2.0 * (x1 * x1 + x2 * x2)),
        ], axis=-1)
    elif order == 'xzy':
        return np.concatenate([
            np.arctan2(2.0 * (x1 * x0 - x2 * x3), -x1 * x1 + x2 * x2 - x3 * x3 + x0 * x0),
            np.arctan2(2.0 * (x2 * x0 - x1 * x3), x1 * x1 - x2 * x2 - x3 * x3 + x0 * x0),
            np.arcsin(np.clip(2.0 * (x1 * x2 + x3 * x0), -1.0, 1.0))
        ], axis=-1)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)


def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum(y[i] * y[i - 1], axis=-1)
        d1 = np.sum(-y[i] * y[i - 1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y


def to_xform(x):
    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., np.newaxis, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., np.newaxis, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., np.newaxis, :],
    ], axis=-2)


def from_euler(e, order='zyx'):
    axis = {'x': np.array([1, 0, 0], dtype=np.float32),
            'y': np.array([0, 1, 0], dtype=np.float32),
            'z': np.array([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))


def from_xform(ts, eps=1e-10):
    qs = np.empty_like(ts[..., :1, 0].repeat(4, axis=-1))

    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[..., np.newaxis],
        (s * (ts[..., 2, 1] - ts[..., 1, 2]))[..., np.newaxis],
        (s * (ts[..., 0, 2] - ts[..., 2, 0]))[..., np.newaxis],
        (s * (ts[..., 1, 0] - ts[..., 0, 1]))[..., np.newaxis]
    ], axis=-1), qs)

    c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)[..., np.newaxis],
        (s0 * 0.25)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)[..., np.newaxis]
    ], axis=-1), qs)

    c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c1)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)[..., np.newaxis],
        (s1 * 0.25)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)[..., np.newaxis]
    ], axis=-1), qs)

    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    qs = np.where(((t <= 0) & c2)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)[..., np.newaxis],
        (s2 * 0.25)[..., np.newaxis]
    ], axis=-1), qs)

    return qs


def fk(lrot, lpos, parents):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def fk_vel(lrot, lpos, lvrt, lvel, parents):
    gp, gr, gt, gv = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvrt[..., :1, :]], [lvel[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))
        gt.append(gt[parents[i]] + mul_vec(gr[parents[i]], lvrt[..., i:i + 1, :]))
        gv.append(gv[parents[i]] + mul_vec(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  _fast_cross(gt[parents[i]], mul_vec(gr[parents[i]], lpos[..., i:i + 1, :])))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2), np.concatenate(gt, axis=-2), np.concatenate(gv,
                                                                                                                 axis=-2)
    



        





import numpy as np
import scipy.signal as sps

from .logs import get_logger_from_arg
from .signal_manipulation import preemphasis


def extract_mel_spectrogram_for_tts(wav_signal, fs, n_fft, step_size, n_mels, mel_fmin, mel_fmax, min_amplitude,
                                    pre_emphasis=True, pre_emph_coeff=0.97, dynamic_range=None, real_amplitude=True,
                                    centered=True, normalize_mel_bins=True, normalize_range=True, logger=None):
    """ Extract mel-spectrogram from an audio signal for TTS training

    :param wav_signal:          Numpy array of audio samples -- shape = (T, )
    :param fs:                  sampling frequency of the audio signal
    :param n_fft:               filter length (in samples) of the FFT
    :param step_size:           length (in samples) between successive analysis windows
    :param n_mels:              number of mel components in the mel-spectrogram
    :param mel_fmin:            minimum frequency used when converting to mel
    :param mel_fmax:            maximum frequency used when converting to mel
    :param min_amplitude:       mel-spectrogram minimal permitted amplitude value (limits the dynamic range)
    :param pre_emphasis:        perform pre-emphasis on input audio
    :param pre_emph_coeff:      pre-emphasis coefficient
    :param dynamic_range:       mel-spectrogram maximal dynamic range in dB (ignored if min_amplitude is specified)
    :param real_amplitude:      if True, the value of the spectrogram bins will be divided by n_fft to get bin magnitude that
                                reflect the temporal signal amplitude
    :param centered:            if True, the spectrogram extraction window will be centered on the time step.
                                The time sequence has to be padded.
    :param normalize_mel_bins:  normalize energy per bins in the mel-spectrogram
    :param normalize_range:     If True, map the db_dynamic_range to [0,1]
    :param logger:              arg to create logger object

    :return: the mel-spectrogram corresponding to the input audio
    """
    # perform pre-emphasis on input audio
    if pre_emphasis:
        wav_signal = preemphasis(wav_signal, preemph=pre_emph_coeff)

    # get linear amplitude spectrogram
    s, _ = extract_spectrogram(x=wav_signal, n_fft=n_fft, step_size=step_size,
                               real_amplitude=real_amplitude, centered=centered)

    # convert to mel frequency scale
    s = linear_to_mel(linear_spectrogram=s, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                      normalize_mel_bins=normalize_mel_bins, logger=logger)

    # extract min amplitude to clip the mel-spectrogram and set the dynamic range
    if min_amplitude or dynamic_range:
        min_amplitude = get_spectrogram_min_amplitude(real_amplitude=real_amplitude, min_amplitude=min_amplitude,
                                                      dynamic_range=dynamic_range, n_fft=n_fft, logger=logger)

    # convert to dB and normalize range to [0, 1]
    s = amplitude_to_db(spectrogram=s, min_amplitude=min_amplitude, normalize_range=normalize_range, logger=logger)

    return s, wav_signal


def get_spectrogram_min_amplitude(real_amplitude, min_amplitude=None, dynamic_range=None, n_fft=None, logger=None):
    """ Compute the minimum amplitude value a spectrogram bin can reach

    :param real_amplitude:  If True, assume that the values of the spectrogram bins were divided by n_fft to get
                            bin magnitude that reflect the temporal signal amplitude
    :param min_amplitude:   The spectrogram minimal permitted amplitude value (limits the dynamic range)
                            This value is affected when real_amplitude is set to True
    :param dynamic_range:   The spectrogram maximal dynamic range in dB (ignored if min_amplitude is specified)
                            This value is affected when real_amplitude is set to True
    :param n_fft:           Number of samples of the FFT window used to extract spectrogram
                            Only used when real_amplitude is set to True
    :param logger:          arg to create logger object

    :return: the minimum amplitude of spectrogram bins
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    if min_amplitude and dynamic_range:
        logger.warning(f'Both "min_amplitude" and "dynamic_range" are specified, '
                       f'only "min_amplitude" ({min_amplitude}) will be considered')
    else:
        assert (min_amplitude or dynamic_range), logger.error(f'Neither "min_amplitude" nor "dynamic_range" are set')

    if real_amplitude:
        assert (n_fft is not None), logger.error(f'"real_amplitude" is set to True but "n_fft" has no value')
    else:
        n_fft = 1  # equivalent to using a FFT window of 1

    if min_amplitude:
        # compute real min amplitude per bin
        min_amplitude = min_amplitude / n_fft

    elif dynamic_range:
        # compute real dynamic range per bin
        dynamic_range = dynamic_range + 20 * np.log10(n_fft)
        # compute real min amplitude per bin
        min_amplitude = 10 ** (-dynamic_range / 20)

    return min_amplitude


def amplitude_to_db(spectrogram, min_amplitude=None, normalize_range=False, logger=None):
    """ Transform amplitude to dB with optional clipping and dynamic range normalization

    :param spectrogram:         Numpy array containing all amplitudes of a spectrogram
    :param min_amplitude:       Clip the spectrogram to the minimal permitted amplitude value
    :param normalize_range:     If True, map the db_dynamic_range to [0,1]
    :param logger:              arg to create logger object

    :return: spectrogram in dB
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # make sure amplitude bins are positive
    spectrogram = np.abs(spectrogram)

    if min_amplitude:
        # apply clipping
        spectrogram = np.clip(spectrogram, a_min=min_amplitude, a_max=None)

    # transform to dB
    spectrogram = 20 * np.log10(spectrogram)

    # normalize range if necessary
    if normalize_range:
        # min_amplitude must be given to normalize de dB dynamic range
        assert (min_amplitude), logger.error(f'Asked for dynamic range normalization, but "min_amplitude" has no value')

        # compute dB dynamic range and map it to [0, 1]
        dynamic_range = -20 * np.log10(min_amplitude)
        spectrogram = (spectrogram + dynamic_range) / dynamic_range

    return spectrogram


def denormalize_range(spectrogram, min_amplitude_used):
    """ Take a dB spectrogram that has been mapped between [0, 1] and shape it back to its original dB dynamic range

    :param spectrogram:         Numpy array containing all amplitudes of a spectrogram in dB (values between 0 and 1)
    :param min_amplitude_used:  Minimal amplitude value that was used to normalize the dB spectrogram dynamic range

    :return: spectrogram in dB with its range de-normalized
    """
    # compute dB dynamic range
    dynamic_range = -20 * np.log10(min_amplitude_used)

    # denormalize dB dynamic range
    spectrogram = spectrogram * dynamic_range - dynamic_range

    return spectrogram


def db_to_amplitude(spectrogram):
    """ Transform dB spectrogram to amplitude spectrogram

     :param spectrogram:    Numpy array containing all amplitude of a spectrogram

     :return: spectrogram in amplitude value
     """
    return 10 ** (spectrogram / 20)


def linear_to_mel(linear_spectrogram, fs=None, n_mels=80, mel_fmin=0, mel_fmax=None,
                  normalize_mel_bins=True, logger=None):
    """ Convert a linear spectrogram to a mel-spectrogram

    :param linear_spectrogram:    Numpy array containing all amplitudes of a spectrogram -- shape = (n_fft // 2 + 1, T)
    :param fs:                    Sampling frequency expected by the algorithm
    :param n_mels:                Number of bins in the mel-spectrogram
    :param mel_fmin:              Lowest frequency in the mel-spectrum (Hz)
    :param mel_fmax:              Highest frequency in the mel-spectrum (Hz)
    :param normalize_mel_bins:    normalize energy per bins in the mel-spectrogram
    :param logger:                arg to create logger object

    :return: Numpy array containing the spectrogram in mel frequency space -- shape = (n_mels, T)
    """
    # find the number of samples used in the FFT window
    n_fft = (linear_spectrogram.shape[0] - 1) * 2

    # get filter parameters
    mel_basis = _get_mel_filterbank_matrix(n_fft=n_fft, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                           normalize_mel_bins=normalize_mel_bins, logger=logger)

    # apply filter bank matrix
    return np.dot(mel_basis, linear_spectrogram)


def mel_to_linear(mel_spectrogram, fs, n_fft, mel_fmin=0, mel_fmax=None, normalize_mel_bins=False, logger=None):
    """ Convert a mel-spectrogram to a linear spectrogram

    :param mel_spectrogram:         Numpy array of the input mel spectrogram -- shape = (n_mels, T)
    :param fs:                      sampling frequency
    :param n_fft:                   number of samples used in the original FFT
    :param mel_fmin:                minimum frequency used when converting to mel
    :param mel_fmax:                maximum frequency used when converting to mel
    :param normalize_mel_bins:      normalize energy per bins in the mel-spectrogram
    :param logger:                  arg to create logger object

    :return: Numpy array containing the spectrogram in linear frequency space -- shape = (n_fft // 2 + 1, T)
    """
    # find the number of mel components
    n_mels = mel_spectrogram.shape[0]

    # get filter parameters
    mel_basis = _get_mel_filterbank_matrix(n_fft=n_fft, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                           normalize_mel_bins=normalize_mel_bins, logger=logger)

    # normalise the row of the mel_basis
    weight_value = mel_basis.sum(axis=1)
    mel_basis = np.divide(mel_basis, weight_value.reshape(n_mels, 1))

    # apply the inverse of the mel_filter bank to the algorithm
    linear_spectrogram = np.dot(np.transpose(mel_spectrogram), mel_basis)

    return np.transpose(linear_spectrogram)


def extract_spectrogram(x, n_fft, step_size, real_amplitude=True, centered=True):
    """ Extract the FFT spectrogram from a series of samples

    :param x:                   Numpy array of input samples -- shape = (T, )
    :param n_fft:               number of point in the FFT window
    :param step_size:           number of samples skipped at each extraction
    :param real_amplitude:      if True the value of the bins will be divided by n_fft to get bin magnitude that
                                reflect the temporal signal amplitude
    :param centered:            if True, the extraction window will be centered on the time step.
                                The time sequence has to be padded.

    :return: Numpy arrays of amplitude and phase of the spectrogram -- shapes = (n_fft // 2 + 1, L)
    """
    # create the sampling window
    window = sps.hann(n_fft)

    # check input signal has a length superior or equal to n_fft
    if len(x) < n_fft:
        x = np.pad(x, (0, len(window) - len(x)), 'constant', constant_values=(0, 0))

    # pad before and after to center the window on the extracted values
    if centered:
        padding_left, padding_right = _get_padding_for_centered_spectrogram(n_fft=n_fft)
        x = np.pad(x, (padding_left, padding_right), mode='reflect')

    # count the number of frames 
    if len(x) % step_size == 0:
        time_axis = int(np.floor((len(x) - n_fft) / step_size))
    else:
        time_axis = 1 + int(np.floor((len(x) - n_fft) / step_size))

    # create container for spectrogram
    amp = np.zeros((n_fft // 2 + 1, time_axis))
    phase = np.zeros((n_fft // 2 + 1, time_axis))

    for i in range(time_axis):
        # get slice of data
        win_data = x[i * step_size: i * step_size + n_fft]

        # apply windowing
        win_data = np.multiply(win_data, window)

        # get FFT
        freq = np.fft.rfft(win_data)

        # save magnitude and phase individually
        amp[:, i] = np.absolute(freq)
        phase[:, i] = np.angle(freq)

    # scale amplitude bins if necessary
    if real_amplitude:
        amp = amp / n_fft

    return amp, phase


def get_nb_spectrogram_samples(wav_length, n_fft, step_size, centered=True):
    """ Return the number of spectrogram time frames given a WAV segment

    :param wav_length:      number of samples in the WAV segment
    :param n_fft:           filter length (in samples) of the FFT
    :param step_size:       length (in samples) between successive analysis windows
    :param centered:        if True, assume that the FFT extraction window is centered on the time step

    :return: the number of spectrogram time frames
    """
    # create random signal
    random_signal = np.random.rand(wav_length)

    # extract amp and phase spectrograms -- shapes = (n_fft // 2 + 1, T)
    amp, phase = extract_spectrogram(x=random_signal, n_fft=n_fft, step_size=step_size, centered=centered)

    # return T
    return amp.shape[1]


def get_nb_wav_samples(spectrogram_length, n_fft, step_size, centered=True):
    ''' Return the number of WAV samples given a spectrogram segment

    :param spectrogram_length:      number of time frames in the spectrogram segment
    :param n_fft:                   filter length (in samples) of the FFT
    :param step_size:               length (in samples) between successive analysis windows
    :param centered:                if True, assume that the FFT extraction window is centered on the time step

    :return: the number of WAV samples
    '''
    # audio segment was padded on the left and right to center the window on the extracted values
    if centered:
        padding_left, padding_right = _get_padding_for_centered_spectrogram(n_fft=n_fft)
    else:
        padding_left, padding_right = 0, 0

    return (spectrogram_length - 1) * step_size + n_fft - padding_left - padding_right


def reconstruct_signal_griffin_lim(magnitude_spectrogram, step_size, iterations=30, logger=None):
    """ Reconstruct an audio signal from a magnitude spectrogram

        Given a magnitude spectrogram as input, reconstruct the audio signal and return it using
        the Griffin-Lim algorithm
        From the paper: "Signal estimation from modified short-time fourier transform" by Griffin and Lim, in IEEE
                        transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    :param magnitude_spectrogram:   Numpy array magnitude spectrogram -- shape = (n_fft // 2 + 1, T)
                                    The rows correspond to frequency bins and the columns correspond to time slices
    :param step_size:               length (in samples) between successive analysis windows
    :param iterations:              Number of iterations for the Griffin-Lim algorithm
                                    Typically a few hundred is sufficient
    :param logger:                  arg to create logger object

    :return: the reconstructed time domain signal as a 1-dim Numpy array and the spectrogram that was used
             to produce the signal
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # shape = (T, n_fft // 2 + 1)
    magnitude_spectrogram = np.transpose(magnitude_spectrogram)

    # find the number of samples used in the FFT window and extract the time steps
    n_fft = (magnitude_spectrogram.shape[1] - 1) * 2
    time_slices = magnitude_spectrogram.shape[0]

    # compute the number of samples needed
    len_samples = int(time_slices * step_size + n_fft)

    # initialize the reconstructed signal to noise
    x_reconstruct = np.random.randn(len_samples)
    window = np.hanning(n_fft)
    n = iterations  # number of iterations of Griffin-Lim algorithm

    while n > 0:
        # decrement and compute FFT
        n -= 1
        reconstruction_spectrogram = np.array([np.fft.rfft(window * x_reconstruct[i: i + n_fft])
                                               for i in range(0, len(x_reconstruct) - n_fft, step_size)])

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * np.angle(reconstruction_spectrogram))

        # store previous reconstructed signal and create a new one by iFFT
        prev_x = x_reconstruct
        x_reconstruct = np.zeros(len_samples)

        for i, j in enumerate(range(0, len(x_reconstruct) - n_fft, step_size)):
            x_reconstruct[j: j + n_fft] += window * np.real(np.fft.irfft(proposal_spectrogram[i]))

        # normalise signal due to overlap add
        x_reconstruct = x_reconstruct / (n_fft / step_size / 2)

        # compute diff between two signals and report progress
        diff = np.sqrt(sum((x_reconstruct - prev_x) ** 2) / x_reconstruct.size)
        logger.debug(f'Reconstruction iteration: {iterations - n}/{iterations} -- RMSE: {diff * 1e6:.3f}e-6')

    return x_reconstruct, proposal_spectrogram


def _get_padding_for_centered_spectrogram(n_fft):
    """ Return padding that must be added to the left and right sides of a series of samples to extract a centered FFT

    :param n_fft:       filter length (in samples) of the FFT

    :return: padding values for left and right sides
    """
    # add same padding on left and right sides
    padding_left, padding_right = int(n_fft // 2), int(n_fft // 2)

    return padding_left, padding_right


def _get_mel_filterbank_matrix(n_fft=None, fs=None, n_mels=80, mel_fmin=0, mel_fmax=None,
                               normalize_mel_bins=False, logger=None):
    """ Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :param n_fft:               number of FFT components
    :param fs:                  sampling rate of the incoming signal
    :param n_mels:              number of Mel bands to generate
    :param mel_fmin:            lowest frequency (in Hz)
    :param mel_fmax:            highest frequency (in Hz). If None, mel_fmax = sr / 2.0
    :param normalize_mel_bins:  normalize energy per bins
    :param logger:              arg to create logger object

    :return: np.ndarray [shape=(n_mels, 1 + n_fft // 2)] -- Mel transform matrix
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # set mel_fmax
    if mel_fmax is None:
        mel_fmax = float(fs) / 2

    # Initialize the weights
    weights = np.zeros((int(n_mels), int(1 + n_fft // 2)))

    # Get the center frequencies of each FFT bin
    fft_freqs = np.linspace(0, float(fs) / 2, int(1 + n_fft // 2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(mel_fmin)
    max_mel = _hz_to_mel(mel_fmax)

    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = _mel_to_hz(mels)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if normalize_mel_bins:  # Normalize energy per bins
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):  # This means we have an empty channel somewhere
        # create logger object (only if needed)
        logger = get_logger_from_arg(logger)
        logger.warning('Empty filters detected in mel frequency basis. Some channels will produce empty responses. '
                       'Try increasing your sampling rate (and fmax) or reducing n_mels.')

    return weights


def _hz_to_mel(frequencies):
    """ Convert Hz to Mels

    :param frequencies:     number or np.ndarray [shape=(n,)] -- scalar or array of frequencies

    :return: number or np.ndarray [shape=(n,)] -- input frequencies in Mels
    """
    # create frequencies array
    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    log_step = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:  # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / log_step

    elif frequencies >= min_log_hz:  # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / log_step

    return mels


def _mel_to_hz(mels):
    """ Convert mel bin numbers to frequencies

    :param mels:    number or np.ndarray [shape=(n,)] -- scalar or array of mel bins to convert

    :return: number or np.ndarray [shape=(n,)] -- input mels in Hz
    """
    # create mels array
    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    log_step = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:  # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(log_step * (mels[log_t] - min_log_mel))

    elif mels >= min_log_mel:  # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(log_step * (mels - min_log_mel))

    return freqs


def pre_emphasis_on_mel(mel_spec, preemph, fs, n_mels, mel_fmin=0, mel_fmax=None, min_amplitude=None,
                        normalized_range=True, logger=''):
    logger = get_logger_from_arg(logger)

    # set mel_fmax
    if mel_fmax is None:
        mel_fmax = float(fs) / 2

    #### get the center frequency of all bins in the mel spectrum ####
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(mel_fmin)
    max_mel = _hz_to_mel(mel_fmax)

    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    bin_freqs = _mel_to_hz(mels)

    #### get the the frequency response of the filter
    a = [1]
    b = [1, -preemph]
    w, h = sps.freqz(b=b, a=a, worN=bin_freqs[1:-1], fs=fs)

    #### apply filter to bins ###
    h = 20 * np.log10(np.abs(h))  # get the filter response in dB
    h = np.tile(np.expand_dims(h, axis=1), (1, mel_spec.shape[1]))

    # if range was normalized
    if normalized_range:
        dbr = -20 * np.log10(min_amplitude)
        # normalize filter
        h = h / dbr

    # Crazy empirical correction hack with magic numbers
    if min_amplitude == 1e-5 and preemph == 0.97:
        correction_matrix = np.log(w) / 30 - 0.277
        correction_matrix = np.tile(np.expand_dims(correction_matrix, axis=1), (1, h.shape[1]))
        h = h - correction_matrix
    else:
        logger.warn("You should probably compute a correction matrix for this config to compensate for the cliping.")

    return np.add(mel_spec, h)

import logging
import os
import sys

from copy import deepcopy
from dateutil.relativedelta import relativedelta


def _get_logger(path_to_file, console_level=logging.DEBUG, file_level=logging.WARNING):
    """ Create a logger object to write in the console and in a file

    :param path_to_file:        name of the log file
    :param console_level:       logging level to write in the console
    :param file_level:          logging level to write in the file

    :return: logger object
    """
    # create folder for log file
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    # create logger
    logger = logging.getLogger(path_to_file)
    logger.setLevel(logging.DEBUG)

    # make it display in the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    formatter = logging.Formatter('%(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # make it write on file
    f_handler = logging.FileHandler(path_to_file)
    f_handler.setLevel(file_level)
    formatter_f = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    f_handler.setFormatter(formatter_f)
    logger.addHandler(f_handler)

    return logger


def _get_root_logger(console_level=logging.DEBUG):
    """ Create a logger object to write in the console

    :param console_level:   logging level to write in the console

    :return: logger object
    """
    # create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # make it display in the console
    console_handler = logging.StreamHandler(sys.stdout)

    # check if there is already one stream handler
    already_exist = False
    for l in logger.handlers:
        if type(console_handler) == type(l):
            already_exist = True

    if not already_exist:
        console_handler.setLevel(console_level)
        formatter = logging.Formatter('(root_logger) [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def _format_logging_level_arg(log_level):
    """ Convert logging level arg to its corresponding int logging level

    :param log_level:   arg referring to the desired logging level

    :return: int corresponding to the desired logging level
    """
    if not isinstance(log_level, str):
        if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            print('LOGGING LEVEL NOT RECOGNIZED -- Setting it to DEBUG')
            return logging.DEBUG
        else:
            return log_level

    else:
        if log_level.lower().strip() == 'debug':
            return logging.DEBUG
        elif log_level.lower().strip() == 'info':
            return logging.INFO
        elif log_level.lower().strip() in ['warn', 'warning']:
            return logging.WARNING
        elif log_level.lower().strip() == 'error':
            return logging.ERROR
        else:
            print('LOGGING LEVEL NOT RECOGNIZED -- Setting it to DEBUG')
            return logging.DEBUG


def get_logger_from_arg(logger=None, console_level=logging.DEBUG, file_level=logging.WARNING):
    """ Create logger instance to display information

    :param logger:          either None, string or logger instance
                            can also be a dict of keywords: {'logger': .., console_level: .., file_level: ..}
    :param console_level:   (string or int) logging level to write in the console
    :param file_level:      (string or int) logging level to write in a file

    :return: logger object
    """
    # check if keyword arguments were passed into a dictionary
    if isinstance(logger, dict):
        logger = get_logger_from_arg(**logger)

    # directly check if logger already exists
    elif not isinstance(logger, type(logging.getLogger())) and not isinstance(logger, type(logging.getLogger('dummy'))):
        # convert arguments to logging.level (ints)
        console_level = _format_logging_level_arg(console_level)
        file_level = _format_logging_level_arg(file_level)

        # if the logger is an empty string, return the console print logger
        if logger == '' or logger is None:
            logger = ConsolePrintLogger(console_level)

        # if logger is a string, it is assumed to be the path of the logger object
        elif isinstance(logger, str):
            logger = _get_logger(logger, console_level, file_level)

        # if it is a fake logger continue
        elif isinstance(logger, FakeLogger) or isinstance(logger, ConsolePrintLogger):
            pass

        # default logger
        else:
            logger = ConsolePrintLogger()

    return logger


def get_args_from_logger(logger):
    """ Retrieve args that were given to create logger object

    :param logger:  logger object that was created using get_logger_from_arg()

    :return: dictionary -- {'logger': .., 'console_level': .., 'file_level': ..}
    """
    # set default values
    kwargs = {'logger': None, 'console_level': logging.DEBUG, 'file_level': logging.WARNING}

    # iterate over handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            kwargs['file_level'] = handler.level
            kwargs['logger'] = handler.baseFilename

        elif isinstance(handler, logging.StreamHandler):
            kwargs['console_level'] = handler.level

    return kwargs


def prog_bar(i, n, bar_size=16):
    """ Create a progress bar to estimate remaining time

    :param i:           current iteration
    :param n:           total number of iterations
    :param bar_size:    size of the bar

    :return: a visualisation of the progress bar
    """
    bar = ''
    done = (i * bar_size) // n

    for j in range(bar_size):
        bar += '█' if j <= done else '░'

    message = f'{bar} {i}/{n}'
    return message


def estimate_required_time(nb_items_in_list, current_index, time_elapsed, interval=100):
    """ Compute a remaining time estimation to process all items contained in a list

    :param nb_items_in_list:        all list items that have to be processed
    :param current_index:           current list index, contained in [0, nb_items_in_list - 1]
    :param time_elapsed:            time elapsed to process current_index items in the list
    :param interval:                estimate remaining time when (current_index % interval) == 0

    :return: time elapsed since the last time estimation
    """
    current_index += 1  # increment current_idx by 1
    if current_index % interval == 0 or current_index == nb_items_in_list:
        # make time estimation and put to string format
        seconds = (nb_items_in_list - current_index) * (time_elapsed / current_index)
        time_estimation = relativedelta(seconds=int(seconds))
        time_estimation_string = f'{time_estimation.hours:02}:{time_estimation.minutes:02}:{time_estimation.seconds:02}'

        # extract progress bar
        progress_bar = prog_bar(i=current_index, n=nb_items_in_list)

        # display info
        if current_index == nb_items_in_list:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- Finished!')
        else:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string}')


def simple_table(item_tuples, logger=None):
    """ Display tuple items in a table

    :param item_tuples:     items to display. Each tuple item is composed of two components (heading, cell)
    :param logger:          arg to create logger object
    """
    # get logger object
    logger = get_logger_from_arg(logger)

    # initialize variables
    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    # extract table items
    headings, cells, = [], []
    for item in item_tuples:
        # extract heading and cell
        heading, cell = str(item[0]), str(item[1])

        # create padding
        pad_head = True if len(heading) < len(cell) else False
        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]
        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:  # pad heading
            heading = pad_left + heading + pad_right
        else:  # pad cell
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    # create the table
    border, head, body = '', '', ''
    for i in range(len(item_tuples)):
        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    # display the table
    logger.info(border)
    logger.info(head)
    logger.info(border)
    logger.info(body)
    logger.info(border)
    logger.info(' ')


def get_all_handler_parameters_from_logger(logger):
    """ Extract handler parameters from a logger object

    :param logger:      logger object

    :return: list of handler parameters contained in logger object
    """
    # initialize list
    handler_list = list()

    # iterate over handler parameters
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            handler_list.append({'type': type(h),
                                 'level': h.level,
                                 'file_name': h.baseFilename,
                                 'format': deepcopy(h.formatter),
                                 'object': None})
        else:
            handler_list.append({'type': type(h),
                                 'level': h.level,
                                 'format':deepcopy(h.formatter),
                                 'object': None})

    return handler_list


class FakeLogger:
    """
        FakeLogger is used in multi-processed functions. It replaces a normal logger object.
        It packages message and send them to a queue that a listener will read and write in a proper logger object.
    """
    # class variables
    queue = None
    propagate = False
    level = 0

    def __init__(self, queue):
        self.queue = queue

    def send_fake_message_on_queue(self, level, msg):
        self.queue.put((level, msg))

    def send_warning_of_fake_logger(self, msg):
        msg = f'Logging when using LaFAT multiprocess facilities disables some logger functionality such as: {msg}'
        self.send_fake_message_on_queue(level=logging.WARNING, msg=msg)

    def critical(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.CRITICAL, msg=message)

    def error(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.ERROR, msg=message)

    def warn(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.WARNING, msg=message)

    def warning(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.WARNING, msg=message)

    def info(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.INFO, msg=message)

    def debug(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.DEBUG, msg=message)

    def log(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=self.level, msg=message)

    def exception(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.ERROR, msg=message)

    def handle(self, record):
        self.queue.put(record)

    def setLevel(self, level):
        self.level = level

    def isEnabledFor(self, level):
        self.send_warning_of_fake_logger(f'isEnabledFor({level})')
        return True

    def getEffectiveLevel(self):
        self.send_warning_of_fake_logger(f'getEffectiveLevel()')
        return 0

    def getChild(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'getChild()')
        return None

    def addFilter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'addFilter()')

    def removeFilter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'removeFilter()')

    def filter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'filter()')

    def addHandler(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'addHandler()')

    def removeHandler(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'removeHandler()')

    def findCaller(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'findCaller()')

    def makeRecord(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'makeRecord()')

    def hasHandlers(self):
        return False


class ConsolePrintLogger:
    """
        FakeLogger is used in multi-processed functions. It replaces a normal logger object.
        It packages message and send them to a queue that a listener will read and write in a proper logger object.
    """
    # class variables
    level = 0

    def __init__(self, level=0):
        self.level = level

    def critical(self, message, *args, **kwargs):
        if self.level <= logging.CRITICAL:
            print(f"[CRITICAL]: {message}")

    def error(self, message, *args, **kwargs):
        if self.level <= logging.ERROR:
            print(f"[ERROR]: {message}")

    def warn(self, message, *args, **kwargs):
        if self.level <= logging.WARNING:
            print(f"[WARNING]: {message}")

    def warning(self, message, *args, **kwargs):
        if self.level <= logging.WARNING:
            print(f"[WARNING]: {message}")

    def info(self, message, *args, **kwargs):
        if self.level <= logging.INFO:
            print(f"[INFO]: {message}")

    def debug(self, message, *args, **kwargs):
        if self.level <= logging.DEBUG:
            print(f"[DEBUG]: {message}")

    def log(self, message, *args, **kwargs):
        print(f"[LOG]: {message}")

    def exception(self, message, *args, **kwargs):
        print(f"[EXCEPTION]: {message}")

    def handle(self, record):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLES")

    def setLevel(self, level):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED LEVELS")

    def isEnabledFor(self, level):
        return True

    def getEffectiveLevel(self):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED LEVELS")
        return 0

    def getChild(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE CHILDS")
        return None

    def addFilter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def removeFilter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def filter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def addHandler(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")

    def removeHandler(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")

    def findCaller(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED CALLERS")

    def makeRecord(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED RECORDS")

    def hasHandlers(self):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")

import os, platform, subprocess

# setup dependencies on import
# check for any missing that the user needs to install and inform them
if platform.system() == "Windows":
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'sox', 'windows')
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'ffmpeg', 'windows')
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'windows')


elif platform.system() == "Linux":
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'linux')
    try:
        output = subprocess.check_output(['sox', '--version'])
    except Exception as e:
       raise Exception("Error calling sox dependency: " + str(e) + "\n\nHave you installed sox with 'apt-get install sox'?\n")

    try:
        output = subprocess.check_output(['ffmpeg', '-version'])
    except Exception as e:
       raise Exception("Error calling ffmpeg dependency: " + str(e) + "\n\nHave you installed ffmpeg with 'apt-get install ffmpeg'?\n")

elif platform.system() == "Darwin":
    # Skipping because not sure if used at all
    # os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'linux')
    try:
        output = subprocess.check_output(['sox', '--version'])
    except Exception as e:
       raise Exception("Error calling sox dependency: " + str(e) + "\n\nHave you installed sox with 'brew install sox'?\n")

    try:
        output = subprocess.check_output(['ffmpeg', '-version'])
    except Exception as e:
       raise Exception("Error calling ffmpeg dependency: " + str(e) + "\n\nHave you installed ffmpeg with 'brew install ffmpeg'?\n")


else:
    raise Exception("Unsupported platform: {}".format(platform.system()))
import os

import numpy as np
import sox
from scipy.io import wavfile

from .logs import get_logger_from_arg


def reformat_and_trim_wav_file(wav_file, fs, bit_depth, nb_channels, overwrite=True, out_path=None,
                               silence_threshold=0.1, min_silence_duration=0.01, silence_pad=True, logger=None):
    """ Format WAV files with the specified parameters using SoX

    :param wav_file:                WAV file to format (full path)
    :param fs:                      desired sampling frequency of WAV file
    :param bit_depth:               desired bit depth of WAV file
    :param nb_channels:             desired number of channels of WAV file
    :param overwrite:               overwrite existing WAV file with their new version
                                    if not, a folder is created to store the new files
    :param out_path:                path to save reformatted WAV file
                                    only used when overwrite is set to False
    :param silence_threshold:       threshold to detect silences
    :param min_silence_duration:    min silence duration to remove
                                    only used when silence_threshold is superior to 0.
    :param silence_pad:             pad audio with silences at the beginning and the end
    :param logger:                  arg to create logger object
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # ---------- DEAL WITH PATHS ----------

    # normalize and strip path
    initial_path = os.path.normpath(wav_file).strip()

    if overwrite:
        # create a temporary filename
        out_path = os.path.join(os.path.dirname(initial_path),
                                os.path.basename(initial_path).replace('.wav', '_tmp.wav'))
    else:
        if out_path:
            # processed WAV file name
            out_path = os.path.normpath(out_path).strip()
        else:
            # create a folder named processed at the file location
            out_path = os.path.join(os.path.dirname(initial_path), f'processed_{fs}')
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, os.path.basename(initial_path))

    # ---------- REFORMAT FILE WITH SOX ----------

    # create transformer
    tfm = sox.Transformer()

    # remove silences
    if silence_threshold > 0.:
        # remove silence at the beginning
        tfm.silence(location=1, silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration, buffer_around_silence=True)
        # remove silence at the end
        tfm.silence(location=-1, silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration, buffer_around_silence=True)

    # re-sample to desired frequency
    tfm.rate(samplerate=fs, quality='h')

    # convert to desired bit depth and number of channels
    tfm.convert(samplerate=None, n_channels=nb_channels, bitdepth=bit_depth)

    # add short silences at the end and beginning of the file
    if silence_pad:
        tfm.pad(start_duration=0.01, end_duration=0.01)

    # display the applied effects in the logger
    logger.info(f'SoX transformer effects: {tfm.effects_log}')

    # create the output file.
    tfm.build(initial_path, out_path)

    # ---------- CLEAN-UP ----------

    # delete original file and replace by new file
    if overwrite:
        os.remove(initial_path)
        os.rename(out_path, initial_path)


def read_wavfile(file_path, rescale=False, desired_fs=None, desired_nb_channels=None, out_type='float32', logger=None):
    """ Read a WAV file and return the samples in a float32 numpy array

    :param file_path:               path to the file to read
    :param rescale:                 rescale the file to get amplitudes in the range between -1 and +1
                                    only the range is rescaled, not the amplitude
    :param desired_fs:              frequency expected from the WAV file
                                    if not specified, the original WAV file sampling frequency is used
    :param desired_nb_channels:     number of channels expected from the WAV file
                                    if not specified, the original WAV number of channels is used
    :param out_type:            desired output type of the audio waveform
    :param logger:                  arg to create logger object

    :return: sampling frequency and samples
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # check arguments make sense
    assert ('int' in out_type or 'float' in out_type), \
        logger.error(f'Inconsistent argument: only output of type "int" or "float" are supported, not "{out_type}"')
    if rescale:
        assert ('float' in out_type), logger.error(f'Inconsistent arguments: cannot rescale if out_type={out_type}')

    # normalize and strip path
    file_path = os.path.normpath(file_path).strip()

    try:
        # try to read the wav file
        fs, x = wavfile.read(file_path)

        # raise exception if sampling frequency, bit depth or number of channels are not correct
        current_bit_depth = int(str(x.dtype).replace('int', '').replace('uint', '').replace('float', ''))
        if desired_fs and fs != desired_fs or desired_nb_channels and len(x.shape) != desired_nb_channels:
            raise BadSamplingFrequencyError(f'Format readable but requirements not met -- currently is '
                                            f'{fs}Hz/{current_bit_depth} bits/{len(x.shape)} channels')

    except (ValueError, BadSamplingFrequencyError) as e:
        # create a reformatted temporary version
        tmp_wav = os.path.join(os.path.dirname(file_path),
                               os.path.basename(file_path).replace('.wav', '_tmp.wav'))

        # add default value if nothing is specified
        desired_fs = desired_fs if desired_fs else 22050
        desired_nb_channels = desired_nb_channels if desired_nb_channels else 1

        # infer desired bit depth with desired out_type
        desired_bit_depth = int(out_type.replace('int', '').replace('uint', '').replace('float', ''))

        # reformat
        logger.info(f'{file_path} -- {e}')
        logger.info(f'converting to {desired_fs}Hz/{desired_bit_depth} bits/{desired_nb_channels} channels')
        reformat_and_trim_wav_file(file_path, fs=desired_fs, bit_depth=desired_bit_depth,
                                   nb_channels=desired_nb_channels, overwrite=False, out_path=tmp_wav,
                                   silence_threshold=-1., silence_pad=False, logger=logger)

        # read reformatted file and delete it
        fs, x = wavfile.read(tmp_wav)
        os.remove(tmp_wav)

    # rescale between -1 and 1 in float32
    if rescale:
        x = _rescale_wav_to_float32(x)

    # extract current waveform dtype and check everything is correct
    current_dtype = str(x.dtype)
    if 'int' in current_dtype and 'float' in out_type:
        logger.warning(f'Waveform is "{current_dtype}", converting to "{out_type}" but values will not be in '
                       f'[-1., 1.] -- Use rescale=True to have samples in [-1., 1.]')
    if 'float' in current_dtype:  # sample values are in [-1., 1.]
        assert ('int' not in out_type), logger.error(f'Waveform is "{current_dtype}", cannot convert to "{out_type}"')

    # cast to desired output type
    x = np.asarray(x).astype(out_type)

    return fs, x


def write_wavefile(fileName, pcmData, sampling_rate, out_type='int16'):
    """ write a WAV file from a numpy array

    :param fileName:                path and file name to write to
    :param pcmData:                 The numpy array containing the PCM data
    :param sampling_rate:           the sampling rate of the data
    :param out_type:                desired output type of the audio waveform
    """
    current_dtype = str(pcmData.dtype)
    if 'float' in current_dtype and out_type == 'int16':  # sample values are in [-1., 1.] convert to [-32k, 32k]
        data = pcmData * 2 ** 15
    else:
        data = pcmData

    data = data.astype(out_type)
    wavfile.write(fileName, sampling_rate, data)


def rescale_wav_array(x, desired_dtype='float32'):
    """ Rescale WAV array to a specified dtype

    rescales the samples in the given array from the range of its current dtype
    to the range of the specified dtype.  see ranges by type below...

    float32 samples are assumed to be in the range [-1.0,1.0], otherwise an exception is raised.

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    :param x:               audio array
    :param desired_dtype:   nuympy dtype to rescale to

    :return: the rescaled audio array in float32
    """
    y = _rescale_wav_to_float32(x)
    z = _rescale_wav_from_float32(y, desired_dtype)
    return z


def _rescale_wav_to_float32(x):
    """ Rescale WAV array between -1.f and 1.f based on the current format

    :param x:           audio array

    :return: the rescaled audio array in float32
    """

    # rescale audio array
    y = np.zeros(x.shape, dtype='float32')
    if x.dtype == 'int16':
        y = x / 32768.0
    elif x.dtype == 'int32':
        y = x / 2147483648.0
    elif x.dtype == 'float32' or x.dtype == 'float64':
        max_ampl = np.max(np.abs(x))
        if max_ampl > 1.0:
            raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- '
                             f'max amplitude: {max_ampl}')
        y = x.astype('float32')
    elif x.dtype == 'uint8':
        y = ((x / 255.0) - 0.5) * 2
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")

    return y


def _rescale_wav_from_float32(x, dtype):
    """ Rescale WAV array from between -1.f and 1.f to the provided format/dtype

    :param x:           audio array
    :param dtype:       numpy dtype to scale to

    :return: the rescaled audio array in specified format/dtype
    """

    max_ampl = np.max(np.abs(x))
    if max_ampl > 1.0:
        raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- ' \
                         f'max amplitude: {max_ampl}')

    # rescale audio array
    y = np.zeros(x.shape, dtype=dtype)
    if dtype == 'int16':
        y = x * 32768.0
    elif dtype == 'int32':
        y = x * 2147483648.0
    elif dtype == 'float32' or dtype == 'float64':
        y = x
    elif dtype == 'uint8':
        y = 255.0 * ((x / 2.0) + 0.5)
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")

    # convert numpy array to provided type
    y = y.astype(dtype)

    return y


class BadSamplingFrequencyError(Exception):
    def __init__(self, message):
        self.message = message

import scipy.signal as sps


def preemphasis(x, preemph=0.97):
    ''' Perform high pass filtering on input signal

    :param x:           signal to filter
    :param preemph:     pre-emphasis factor

    :return: high pass filtered signal
    '''
    return sps.lfilter([1, -preemph], [1], x)

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SGDataset(Dataset):
    def __init__(
            self,
            path_data_definition,
            path_processed_data,
            window,
            style_encoding_type,
            example_window_length,
    ):
        """PyTorch Dataset Instance

        Args:
            path_data_definition : Path to data_definition file
            path_processed_data : Path to processed_data npz file
            window : Length of the input-output slice
            style_encoding_type : "label" or "example"
            example_window_length : Length of example window
        """

        with open(path_data_definition, "r") as f:
            details = json.load(f)
        self.details = details
        self.njoints = len(details["bone_names"])
        self.nlabels = len(details["label_names"])
        self.label_names = details["label_names"]
        self.bone_names = details["bone_names"]
        self.parents = torch.LongTensor(details["parents"])
        self.dt = details["dt"]
        self.window = window
        self.style_encoding_type = style_encoding_type
        self.example_window_length = example_window_length

        # Load Data
        processed_data = np.load(path_processed_data)

        self.ranges_train = processed_data["ranges_train"]
        self.ranges_valid = processed_data["ranges_valid"]
        self.ranges_train_labels = processed_data["ranges_train_labels"]
        self.ranges_valid_labels = processed_data["ranges_valid_labels"]

        self.X_audio_features = torch.as_tensor(
            processed_data["X_audio_features"], dtype=torch.float32
        )
        self.Y_root_pos = torch.as_tensor(processed_data["Y_root_pos"], dtype=torch.float32)
        self.Y_root_rot = torch.as_tensor(processed_data["Y_root_rot"], dtype=torch.float32)
        self.Y_root_vel = torch.as_tensor(processed_data["Y_root_vel"], dtype=torch.float32)
        self.Y_root_vrt = torch.as_tensor(processed_data["Y_root_vrt"], dtype=torch.float32)
        self.Y_lpos = torch.as_tensor(processed_data["Y_lpos"], dtype=torch.float32)
        self.Y_ltxy = torch.as_tensor(processed_data["Y_ltxy"], dtype=torch.float32)
        self.Y_lvel = torch.as_tensor(processed_data["Y_lvel"], dtype=torch.float32)
        self.Y_lvrt = torch.as_tensor(processed_data["Y_lvrt"], dtype=torch.float32)
        self.Y_gaze_pos = torch.as_tensor(processed_data["Y_gaze_pos"], dtype=torch.float32)

        self.audio_input_mean = torch.as_tensor(
            processed_data["audio_input_mean"], dtype=torch.float32
        )
        self.audio_input_std = torch.as_tensor(
            processed_data["audio_input_std"], dtype=torch.float32
        )
        self.anim_input_mean = torch.as_tensor(
            processed_data["anim_input_mean"], dtype=torch.float32
        )
        self.anim_input_std = torch.as_tensor(processed_data["anim_input_std"], dtype=torch.float32)
        self.anim_output_mean = torch.as_tensor(
            processed_data["anim_output_mean"], dtype=torch.float32
        )
        self.anim_output_std = torch.as_tensor(
            processed_data["anim_output_std"], dtype=torch.float32
        )

        # Build Windows
        R = []
        L = []
        S = []
        for sample_number, ((range_start, range_end), range_label) in enumerate(
                zip(self.ranges_train, self.ranges_train_labels)
        ):

            one_hot_label = np.zeros(self.nlabels, dtype=np.float32)
            one_hot_label[range_label] = 1.0

            for ri in range(range_start, range_end - window):
                R.append(np.arange(ri, ri + window))
                L.append(one_hot_label)
                S.append(sample_number)

        self.R = torch.as_tensor(np.array(R), dtype=torch.long)
        self.L = torch.as_tensor(np.array(L), dtype=torch.float32)
        self.S = torch.as_tensor(S, dtype=torch.short)
        # self.get_stats()

    @property
    def example_window_length(self):
        return self._example_window_length

    @example_window_length.setter
    def example_window_length(self, a):
        self._example_window_length = a

    def __len__(self):
        return len(self.R)

    def __getitem__(self, index):
        # Extract Windows
        Rwindow = self.R[index]
        Rwindow = Rwindow.contiguous()

        # Extract Labels
        Rlabel = self.L[index]

        # Get Corresponding Ranges for Style Encoding
        RInd = self.S[index]
        sample_range = self.ranges_train[RInd]

        # Extract Audio
        W_audio_features = self.X_audio_features[Rwindow]

        # Extract Animation
        W_root_pos = self.Y_root_pos[Rwindow]
        W_root_rot = self.Y_root_rot[Rwindow]
        W_root_vel = self.Y_root_vel[Rwindow]
        W_root_vrt = self.Y_root_vrt[Rwindow]
        W_lpos = self.Y_lpos[Rwindow]
        W_ltxy = self.Y_ltxy[Rwindow]
        W_lvel = self.Y_lvel[Rwindow]
        W_lvrt = self.Y_lvrt[Rwindow]
        W_gaze_pos = self.Y_gaze_pos[Rwindow]

        if self.style_encoding_type == "label":
            style = Rlabel
        elif self.style_encoding_type == "example":
            style = self.get_example(Rwindow, sample_range, self.example_window_length)

        return (
            W_audio_features,
            W_root_pos,
            W_root_rot,
            W_root_vel,
            W_root_vrt,
            W_lpos,
            W_ltxy,
            W_lvel,
            W_lvrt,
            W_gaze_pos,
            style,
        )

    def get_shapes(self):
        num_audio_features = self.X_audio_features.shape[1]
        pose_input_size = len(self.anim_input_std)
        pose_output_size = len(self.anim_output_std)
        dimensions = dict(
            num_audio_features=num_audio_features,
            pose_input_size=pose_input_size,
            pose_output_size=pose_output_size,
        )
        return dimensions

    def get_means_stds(self, device):
        return (
            self.audio_input_mean.to(device),
            self.audio_input_std.to(device),
            self.anim_input_mean.to(device),
            self.anim_input_std.to(device),
            self.anim_output_mean.to(device),
            self.anim_output_std.to(device),
        )

    def get_example(
            self, Rwindow, sample_range, example_window_length,
    ):

        ext_window = (example_window_length - self.window) // 2
        ws = min(ext_window, Rwindow[0] - sample_range[0])
        we = min(ext_window, sample_range[1] - Rwindow[-1])
        s_ext = ws + ext_window - we
        w_ext = we + ext_window - ws
        start = max(Rwindow[0] - s_ext, sample_range[0])
        end = min(Rwindow[-1] + w_ext, sample_range[1]) + 1
        end = min(end, len(self.Y_root_vel))
        S_root_vel = self.Y_root_vel[start:end].reshape(end - start, -1)
        S_root_vrt = self.Y_root_vrt[start:end].reshape(end - start, -1)
        S_lpos = self.Y_lpos[start:end].reshape(end - start, -1)
        S_ltxy = self.Y_ltxy[start:end].reshape(end - start, -1)
        S_lvel = self.Y_lvel[start:end].reshape(end - start, -1)
        S_lvrt = self.Y_lvrt[start:end].reshape(end - start, -1)
        example_feature_vec = torch.cat(
            [S_root_vel, S_root_vrt, S_lpos, S_ltxy, S_lvel, S_lvrt, torch.zeros_like(S_root_vel), ],
            dim=1,
        )
        curr_len = len(example_feature_vec)
        if curr_len < example_window_length:
            example_feature_vec = torch.cat(
                [example_feature_vec, example_feature_vec[-example_window_length + curr_len:]],
                dim=0,
            )
        return example_feature_vec

    def get_sample(self, dataset, length=None, range_index=None):
        if dataset == "train":
            if range_index is None:
                range_index = np.random.randint(len(self.ranges_train))
            (s, e), label = self.ranges_train[range_index], self.ranges_train_labels[range_index]
        elif dataset == "valid":
            if range_index is None:
                range_index = np.random.randint(len(self.ranges_valid))
            (s, e), label = self.ranges_valid[range_index], self.ranges_valid_labels[range_index]

        if length is not None:
            e = min(s + length * 60, e)

        return (
            self.X_audio_features[s:e][np.newaxis],
            self.Y_root_pos[s:e][np.newaxis],
            self.Y_root_rot[s:e][np.newaxis],
            self.Y_root_vel[s:e][np.newaxis],
            self.Y_root_vrt[s:e][np.newaxis],
            self.Y_lpos[s:e][np.newaxis],
            self.Y_ltxy[s:e][np.newaxis],
            self.Y_lvel[s:e][np.newaxis],
            self.Y_lvrt[s:e][np.newaxis],
            self.Y_gaze_pos[s:e][np.newaxis],
            label,
            [s, e],
            range_index,
        )

    def get_stats(self):
        from rich.console import Console
        from rich.table import Table

        console = Console(record=True)
        # Style infos
        df = pd.DataFrame()
        df["Dataset"] = ["Train", "Validation", "Total"]
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        table = Table(title="Data Info", show_lines=True, row_styles=["magenta"])
        table.add_column("Dataset")
        data_len = 0
        for i in range(self.nlabels):
            ind_mask = self.ranges_train_labels == i
            ranges = self.ranges_train[ind_mask]
            num_train_frames = (
                    np.sum(ranges[:, 1] - ranges[:, 0]) / 2
            )  # It is divided by two as we have mirrored versions too
            ind_mask = self.ranges_valid_labels == i
            ranges = self.ranges_valid[ind_mask]
            num_valid_frames = np.sum(ranges[:, 1] - ranges[:, 0]) / 2
            total = num_train_frames + num_valid_frames
            df[self.label_names[i]] = [
                f"{num_train_frames} frames - {num_train_frames / 60:.1f} secs",
                f"{num_valid_frames} frames - {num_valid_frames / 60:.1f} secs",
                f"{total} frames - {total / 60:.1f} secs",
            ]
            table.add_column(self.label_names[i])
            data_len += total

        for i in range(3):
            table.add_row(*list(df.iloc[i]))
        console.print(table)
        dimensions = self.get_shapes()
        console.print(f"Total length of dataset is {data_len} frames - {data_len / 60:.1f} seconds")
        console.print("Num features: ", dimensions)

