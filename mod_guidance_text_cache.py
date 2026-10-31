import weakref
from collections import OrderedDict

# TE module -> LRU of (text + TE state) -> (raw, t5_ids, t5_weights). Weak on the
# module so the entries go when ComfyUI drops the loaded text encoder.
_TEXT_CACHE = weakref.WeakKeyDictionary()
_TEXT_CACHE_SIZE = 8


def _encode_text_cached(clip, text):
    """Encode `text` through `clip`, reusing the CPU tensors of an earlier run.

    The sampler node executes on every seed, but quality_tags / quality_neg
    rarely change, and ComfyUI's own node cache does not cover an encode made
    inside the sampler. Callers must not write to the returned tensors.
    """
    entries = _TEXT_CACHE.setdefault(clip.cond_stage_model, OrderedDict())
    key = (
        text,
        clip.patcher.patches_uuid,
        clip.layer_idx,
        repr(clip.tokenizer_options),
    )
    cached = entries.get(key)
    if cached is not None:
        entries.move_to_end(key)
        return cached

    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    t5_ids = output.get("t5xxl_ids")
    t5_weights = output.get("t5xxl_weights")
    encoded = (
        output["cond"][0].detach().cpu(),
        t5_ids.detach().cpu() if t5_ids is not None else None,
        t5_weights.detach().cpu() if t5_weights is not None else None,
    )
    entries[key] = encoded
    while len(entries) > _TEXT_CACHE_SIZE:
        entries.popitem(last=False)
    return encoded
