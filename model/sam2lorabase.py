import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from torchvision import transforms
from sam2.modeling.sam2_base import SAM2Base
from model.lora_adapters import ImageEncoderAdapter, MaskDecoderAdapter, PromptEncoderAdapter
from typing import Any, Optional

NO_OBJ_SCORE: float = -1024.0


class SAM2LoRABase(SAM2Base):
    def __init__(self,
                 sam_model: Any,
                 rank: int = 32,
                 alpha: int = 64,
                 dropout: float = 0.1,
                 num_maskmem: int = 7,  # default 1 input frame + 6 previous frames
                 image_size: int = 1024,
                 backbone_stride: int = 16,  # stride of the image backbone output
                 sigmoid_scale_for_mem_enc: float = 1.0,  # scale factor for mask sigmoid prob
                 sigmoid_bias_for_mem_enc: float = 0.0,  # bias factor for mask sigmoid prob
                 # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
                 binarize_mask_from_pts_for_mem_enc: bool = False,
                 use_mask_input_as_output_without_sam: bool = False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
                 # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
                 # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
                 # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
                 max_cond_frames_in_attn: int = -1,
                 # on the first frame, whether to directly add the no-memory embedding to the image feature
                 # (instead of using the transformer encoder)
                 directly_add_no_mem_embed: bool = False,
                 # whether to use high-resolution feature maps in the SAM mask decoder
                 use_high_res_features_in_sam: bool = False,
                 # whether to output multiple (3) masks for the first click on initial conditioning frames
                 multimask_output_in_sam: bool = False,
                 # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
                 # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
                 multimask_min_pt_num: int = 1,
                 multimask_max_pt_num: int = 1,
                 # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
                 multimask_output_for_tracking: bool = False,
                 # Whether to use multimask tokens for obj ptr; Only relevant when both
                 # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
                 use_multimask_token_for_obj_ptr: bool = False,
                 # whether to use sigmoid to restrict ious prediction to [0-1]
                 iou_prediction_use_sigmoid: bool = False,
                 # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
                 # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
                 # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
                 memory_temporal_stride_for_eval: int = 1,
                 # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
                 non_overlap_masks_for_mem_enc: bool = False,
                 # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
                 use_obj_ptrs_in_encoder: bool = False,
                 # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
                 max_obj_ptrs_in_encoder: int = 16,
                 # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
                 add_tpos_enc_to_obj_ptrs: bool = True,
                 # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
                 # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
                 proj_tpos_enc_in_obj_ptrs: bool = False,
                 # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
                 # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
                 use_signed_tpos_enc_to_obj_ptrs: bool = False,
                 # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
                 # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
                 only_obj_ptrs_in_the_past_for_eval: bool = False,
                 # Whether to predict if there is an object in the frame
                 pred_obj_scores: bool = False,
                 # Whether to use an MLP to predict object scores
                 pred_obj_scores_mlp: bool = False,
                 # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
                 # Whether to have a fixed no obj pointer when there is no object present
                 # or to use it as an additive embedding with obj_ptr produced by decoder
                 fixed_no_obj_ptr: bool = False,
                 # Soft no object, i.e. mix in no_obj_ptr softly,
                 # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
                 soft_no_obj_ptr: bool = False,
                 use_mlp_for_obj_ptr_proj: bool = False,
                 # add no obj embedding to spatial frames
                 no_obj_embed_spatial: bool = False,
                 # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
                 sam_mask_decoder_extra_args: Optional[dict] = None,
                 compile_image_encoder: bool = False) -> None:
        super(SAM2LoRABase, self).__init__(
            sam_model.image_encoder,
            sam_model.memory_attention,
            sam_model.memory_encoder,
            num_maskmem,
            image_size,
            backbone_stride,
            sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,
            directly_add_no_mem_embed,
            use_high_res_features_in_sam,
            multimask_output_in_sam,
            multimask_min_pt_num,
            multimask_max_pt_num,
            multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores,
            pred_obj_scores_mlp,
            fixed_no_obj_ptr,
            soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial,
            sam_mask_decoder_extra_args,
            compile_image_encoder)

        # Freeze the SAM model parameters
        for param in sam_model.parameters():
            param.requires_grad = False

        # Extract the encoder (trunk) from SAM
        self.image_encoder = sam_model.image_encoder

        # Apply adapters to image encoder blocks
        self.image_encoder.trunk.blocks = nn.Sequential(
            *[ImageEncoderAdapter(block, rank=rank, alpha=alpha, dropout=dropout, device=self.device)
              for block in self.image_encoder.trunk.blocks]
        )

        # prompt encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder
        self.sam_mask_decoder.transformer = MaskDecoderAdapter(
            self.sam_mask_decoder.transformer, rank=rank, alpha=alpha, dropout=dropout, device=self.device
        )

        # Get all other layers
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        # self.sam_prompt_encoder.mask_downscaling = PromptEncoderAdapter(self.sam_prompt_encoder.mask_downscaling, rank=rank, alpha=alpha, dropout=dropout, device=self.device)
        self.memory_encoder = sam_model.memory_encoder
        self.memory_attention = sam_model.memory_attention
        if use_obj_ptrs_in_encoder:
            self.mask_downsample = sam_model.mask_downsample

        self.obj_ptr_tpos_proj = sam_model.obj_ptr_tpos_proj
        self.obj_ptr_proj = sam_model.obj_ptr_proj

        """The following is from SAM2Base"""
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting

        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together

        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = sam_model.no_obj_embed_spatial
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = sam_model.no_obj_ptr
            trunc_normal_(self.no_obj_ptr, std=0.02)

        # Part 3: memory encoder for the previous frame's outputs
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.no_mem_embed = sam_model.no_mem_embed
        self.no_mem_pos_enc = sam_model.no_mem_pos_enc
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device