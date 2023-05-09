import logging
import torch

from seq2seq_vc.layers.positional_encoding import ScaledPositionalEncoding
from seq2seq_vc.modules.transformer.encoder import Encoder as TransformerEncoder
from seq2seq_vc.modules.conformer.encoder import Encoder as ConformerEncoder
from seq2seq_vc.modules.transformer.decoder import Decoder
from seq2seq_vc.modules.pre_postnets import Prenet, Postnet
from seq2seq_vc.modules.transformer.mask import subsequent_mask
from seq2seq_vc.layers.utils import make_pad_mask, make_non_pad_mask
from seq2seq_vc.modules.transformer.attention import MultiHeadedAttention
from seq2seq_vc.modules.duration_predictor import DurationPredictor
from seq2seq_vc.modules.length_regulator import LengthRegulator


class FastSpeechVC(torch.nn.Module):
    def __init__(
        self,
        idim,
        odim,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_input_layer: str = "linear",
        encoder_input_conv_kernel_size: int = 3,
        encoder_normalize_before: bool = False,
        decoder_normalize_before: bool = False,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        encoder_reduction_factor: int = 1,
        decoder_reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        # only for conformer
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # pretrained spk emb
        spk_embed_dim: int = None,
        spk_embed_integration_type: str = "add",
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        duration_predictor_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        # teacher model related
        # teacher_model_decoder_reduction_factor: int = 1,
    ):
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        self.encoder_reduction_factor = encoder_reduction_factor
        self.decoder_reduction_factor = decoder_reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.encoder_input_layer = encoder_input_layer
        # self.teacher_model_decoder_reduction_factor = teacher_model_decoder_reduction_factor

        # define encoder
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer="conv2d-scaled-pos-enc",
                pos_enc_class=ScaledPositionalEncoding,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,  # V
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,  # V
                dropout_rate=transformer_enc_dropout_rate,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim * encoder_reduction_factor,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
            )
        else:
            raise NotImplementedError

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)
        
        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()
        
        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * decoder_reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

    def _reset_parameters(self, init_enc_alpha: float, init_dec_alpha: float):
        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer":
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer":
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor = None,
        ds: torch.Tensor = None,
        spembs: torch.Tensor = None,
        is_inference: bool = False,
        alpha: float = 1.0,
    ):
        # check encoder reduction factor
        if self.encoder_reduction_factor > 1:
            # reshape inputs if use reduction factor for encoder
            # (B, Tmax, idim) ->  (B, Tmax // r_e, idim * r_e)
            batch_size, max_length, dim = xs.shape
            if max_length % self.encoder_reduction_factor != 0:
                xs = xs[:, : -(max_length % self.encoder_reduction_factor)]
            xs = xs.contiguous().view(
                batch_size,
                max_length // self.encoder_reduction_factor,
                dim * self.encoder_reduction_factor,
            )
            ilens = ilens.new([ilen // self.encoder_reduction_factor for ilen in ilens])

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # adjust ilens if using downsampling conv2d
        if self.encoder_input_layer == "conv2d":
            ilens = ilens.new([((ilen - 2 + 1) // 2 - 2 + 1) // 2 for ilen in ilens])

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = make_non_pad_mask(ilens).to(xs.device)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, alpha)  # (B, Lmax, adim)
        else:
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, ds)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None and not is_inference:
            if self.decoder_reduction_factor > 1:
                olens_in = olens.new(
                    [olen // self.decoder_reduction_factor for olen in olens]
                )
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        # print("hs", hs.shape)
        # print("h_masks", h_masks.shape)
        
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, ilens

    def forward(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: torch.Tensor,
        tgt_speech: torch.Tensor,
        tgt_speech_lengths: torch.Tensor,
        durations: torch.Tensor,
        durations_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
    ):
        """Calculate forward propagation.

        Args:
            src_speech (Tensor): Batch of padded source features (B, Tmax, odim).
            src_speech_lengths (LongTensor): Batch of the lengths of each source (B,).
            tgt_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            tgt_speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, Tmax + 1).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.

        """
        src_speech = src_speech[:, : src_speech_lengths.max()]  # for data-parallel
        tgt_speech = tgt_speech[:, : tgt_speech_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max()]  # for data-parallel

        batch_size = src_speech.size(0)

        xs, ys, ds = src_speech, tgt_speech, durations
        ilens, olens = src_speech_lengths, tgt_speech_lengths

        # forward propagation
        before_outs, after_outs, d_outs, ilens_ = self._forward(
            xs, ilens, olens, ds, spembs=spembs, is_inference=False
        )

        # modifiy mod part of groundtruth
        # if self.encoder_reduction_factor > 1:
            # ilens = ilens.new([ilen // self.encoder_reduction_factor for ilen in ilens])
        if self.decoder_reduction_factor > 1:
            olens = olens.new(
                [olen - olen % self.decoder_reduction_factor for olen in olens]
            )
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        return before_outs, after_outs, d_outs, ilens_, olens, ys

    def inference(
        self,
        src_speech: torch.Tensor,
        tgt_speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        durations: torch.Tensor = None,
        alpha: float = 1.0,
        use_teacher_forcing: bool = False,
    ):
        """Generate the sequence of features given the sequences of characters.

        Args:
            src_speech (Tensor): Source feature sequence (T, idim).
            tgt_speech (Tensor, optional): Target feature sequence (L, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        x, y = src_speech, tgt_speech
        spemb, d = spembs, durations

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds = d.unsqueeze(0)
            _, outs, *_ = self._forward(
                xs,
                ilens,
                ds=ds,
                spembs=spembs,
            )  # (1, L, odim)
        else:
            # inference
            _, outs, d_outs, _ = self._forward(
                xs,
                ilens,
                spembs=spembs,
                is_inference=True,
                alpha=alpha,
            )  # (1, L, odim)

        return outs[0], d_outs[0]

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)