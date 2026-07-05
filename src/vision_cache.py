"""Precomputes and caches frozen vision-encoder patch embeddings per image.

The vision encoder is frozen and deterministic (see FusionVLM.__init__),
so a given image always produces the exact same patch embeddings no matter
how many epochs, connectors, or script invocations touch it. Without this
cache, FusionVLM._vision_features() re-runs the full SigLIP forward pass on
every single training step and every eval sample -- pure wasted compute,
since only the trainable connector actually needs a fresh forward pass.

This precomputes each split's embeddings ONCE (batched, no_grad) and caches
them both to disk (keyed by split dir + vision model id + sample count, so
repeated invocations and every connector in an ablation sweep reuse the same
cache) and hands back an in-memory tensor aligned to dataset indices.
"""

from pathlib import Path

import torch


def cache_path_for(split_dir, vision_model_id: str, n_samples: int) -> Path:
    safe_id = vision_model_id.replace("/", "_")
    return Path(split_dir) / f".vision_features_cache_{safe_id}_n{n_samples}.pt"


@torch.no_grad()
def precompute_vision_features(
    dataset, vision_model, image_processor, device, split_dir, vision_model_id, batch_size=8
):
    """Returns a (N, num_patches, vision_dim) CPU tensor aligned to `dataset`
    indices (`dataset` may be a DocVQADataset or a Subset of one). Loads from
    disk if a matching cache already exists, otherwise computes it once and
    saves it for next time."""
    n = len(dataset)
    cache_path = cache_path_for(split_dir, vision_model_id, n)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.size(0) == n:
            print(f"[vision cache] hit  -> {cache_path} ({n} samples, vision tower skipped)")
            return cached
        print(f"[vision cache] stale (expected {n}, found {cached.size(0)}) -- recomputing")

    was_training = vision_model.training
    vision_model.eval()

    feats_chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        images = [dataset[i]["image"] for i in range(start, end)]
        pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        feats = vision_model.vision_model(pixel_values=pixel_values).last_hidden_state
        feats_chunks.append(feats.cpu())

    vision_model.train(was_training)

    features = torch.cat(feats_chunks, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    print(f"[vision cache] saved -> {cache_path} ({n} samples)")
    return features
