import torch
import torch.nn as nn

from config import IMAGE_TOKEN


class FusionVLM(nn.Module):
    """Frozen vision encoder + frozen LLM tied through a trainable connector.

    Two fusion paths, selected by `connector.fusion_type`:
      - "prefix": connector output tokens replace a single <image> placeholder
        in the input sequence (MLP-concat, Q-Former).
      - "cross_attention": text stream is left intact; gated cross-attention
        blocks are injected into chosen LLM layers via forward hooks and attend
        to projected image features (Flamingo-style)."""

    def __init__(self, vision_model, connector, llm, tokenizer, image_processor):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
        self.llm = llm
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.fusion_type = getattr(connector, "fusion_type", "prefix")

        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            llm.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        for p in self.vision_model.parameters():
            p.requires_grad_(False)
        for p in self.llm.parameters():
            p.requires_grad_(False)

        self._hooks = []
        if self.fusion_type == "cross_attention":
            self._register_cross_attention_hooks()

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def _register_cross_attention_hooks(self):
        for idx in self.connector.inject_layers:
            block = self.connector.blocks[str(idx)]
            handle = self.llm.model.layers[idx].register_forward_hook(
                self._make_hook(block)
            )
            self._hooks.append(handle)

    def _make_hook(self, block):
        def hook(module, inputs, output):
            feats = self.connector.image_features
            if feats is None:
                return output
            if isinstance(output, tuple):
                return (block(output[0], feats),) + tuple(output[1:])
            return block(output, feats)

        return hook

    def _vision_features(self, pixel_values):
        with torch.no_grad():
            return self.vision_model.vision_model(
                pixel_values=pixel_values
            ).last_hidden_state

    # ---- prefix path (mlp_concat, qformer) ----
    def _merge_prefix(self, pixel_values, input_ids_list, labels_list=None):
        device = self.device
        image_embeds = self.connector(self._vision_features(pixel_values.to(device)))
        embed_layer = self.llm.get_input_embeddings()

        merged_embeds, merged_labels = [], []
        for b, ids in enumerate(input_ids_list):
            ids = ids.to(device)
            text_embeds = embed_layer(ids)
            pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0][0].item()
            img = image_embeds[b].to(text_embeds.dtype)
            merged_embeds.append(
                torch.cat([text_embeds[:pos], img, text_embeds[pos + 1 :]], dim=0)
            )
            if labels_list is not None:
                lab = labels_list[b].to(device)
                img_lab = torch.full((img.size(0),), -100, dtype=lab.dtype, device=device)
                merged_labels.append(torch.cat([lab[:pos], img_lab, lab[pos + 1 :]], dim=0))

        max_len = max(e.size(0) for e in merged_embeds)
        dim = merged_embeds[0].size(1)
        batch = len(merged_embeds)
        inputs_embeds = torch.zeros(batch, max_len, dim, device=device, dtype=merged_embeds[0].dtype)
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        labels = torch.full((batch, max_len), -100, dtype=torch.long, device=device) if labels_list is not None else None
        for b, emb in enumerate(merged_embeds):
            length = emb.size(0)
            inputs_embeds[b, :length] = emb
            attention_mask[b, :length] = 1
            if labels_list is not None:
                labels[b, :length] = merged_labels[b]
        return inputs_embeds, attention_mask, labels

    # ---- cross-attention path ----
    def _pad_text(self, input_ids_list, labels_list=None):
        device = self.device
        max_len = max(x.size(0) for x in input_ids_list)
        batch = len(input_ids_list)
        input_ids = torch.full((batch, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long) if labels_list is not None else None
        for b, ids in enumerate(input_ids_list):
            length = ids.size(0)
            input_ids[b, :length] = ids
            attention_mask[b, :length] = 1
            if labels_list is not None:
                labels[b, :length] = labels_list[b]
        labels = labels.to(device) if labels is not None else None
        return input_ids.to(device), attention_mask.to(device), labels

    def forward(self, pixel_values, input_ids_list, labels_list):
        if self.fusion_type == "cross_attention":
            device = self.device
            self.connector.set_image(self._vision_features(pixel_values.to(device)))
            input_ids, attention_mask, labels = self._pad_text(input_ids_list, labels_list)
            out = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            self.connector.clear_image()
            return out

        inputs_embeds, attention_mask, labels = self._merge_prefix(
            pixel_values, input_ids_list, labels_list
        )
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, image, prompt: str, max_new_tokens: int = 32):
        if IMAGE_TOKEN not in prompt:
            prompt = f"{IMAGE_TOKEN}\n{prompt}"
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        if self.fusion_type == "cross_attention":
            self.connector.set_image(self._vision_features(pixel_values.to(self.device)))
            input_ids, attention_mask, _ = self._pad_text([ids])
            out = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            self.connector.clear_image()
            text = self.tokenizer.decode(out[0, input_ids.size(1):], skip_special_tokens=True)
            return text

        inputs_embeds, attention_mask, _ = self._merge_prefix(pixel_values, [ids])
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
