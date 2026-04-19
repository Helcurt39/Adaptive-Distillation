from transformers import AutoTokenizer

def get_collate_fn(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)

    def collate_fn(batch):
        images, reports, labels = zip(*batch)

        import torch
        from torch.utils.data import default_collate

        images = default_collate(images)
        labels = default_collate(labels)

        # tokenize reports as a batch
        encoded = tokenizer(
            list(reports),
            padding=True,
            truncation=True,
            max_length=cfg.text_max_len,
            return_tensors="pt"
        )

        return images, encoded["input_ids"], encoded["attention_mask"], labels

    return collate_fn