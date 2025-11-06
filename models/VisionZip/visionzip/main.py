from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
from .clip_encoder import CLIPVisionTower_VisionZip
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi, restore_image_features_sorted

def visionzip(model, dominant=191, contextual=30):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual)


    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    # Patch the forward methods - CRITICAL: This must happen before any forward passes
    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward
    
    # Also patch all existing instances (critical for already-loaded models)
    # This is necessary because instances might have bound methods
    vision_tower = model.model.vision_tower.vision_tower
    for name, module in vision_tower.named_modules():
        if isinstance(module, CLIPAttention):
            # Patch the instance method directly
            import types
            module.forward = types.MethodType(CLIPAttention_forward, module)
        elif isinstance(module, CLIPEncoderLayer):
            # Patch the instance method directly
            import types
            module.forward = types.MethodType(CLIP_EncoderLayer_forward, module)
    
    # Verify patching worked
    import inspect
    if not inspect.isfunction(CLIPAttention.forward) or CLIPAttention.forward.__name__ != 'CLIPAttention_forward':
        import warnings
        warnings.warn("CLIPAttention.forward patching may have failed!")
    if not inspect.isfunction(CLIPEncoderLayer.forward) or CLIPEncoderLayer.forward.__name__ != 'CLIP_EncoderLayer_forward':
        import warnings
        warnings.warn("CLIPEncoderLayer.forward patching may have failed!")

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    
    # Create a wrapper function that properly binds self
    def visionzip_forward(self, images):
        """Wrapper that calls VisionZip forward with proper self binding"""
        return CLIPVisionTower_VisionZip.forward(self, images)
    
    # Patch the class method - this will work for all instances
    CLIPVisionTower.forward = visionzip_forward

    # Also ensure the instance method is patched (needed for multi-GPU where device_map might create wrapper objects)
    if hasattr(model.model, 'vision_tower'):
        vision_tower = model.model.vision_tower
        # Bind the method to this specific instance
        import types
        vision_tower.forward = types.MethodType(visionzip_forward, vision_tower)

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip


    return model
