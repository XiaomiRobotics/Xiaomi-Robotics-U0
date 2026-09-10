# Data configuration

wm-fsdp trains on pre-tokenized U0 text and native IBQ visual IDs. It does not run a
vision encoder in DataLoader workers.

## Packed records

The release training launchers use fixed-length `.idx` and `.bin` records:

```json
{
  "type": "megatron_packed",
  "format": "multimodal_sft_v1",
  "sequence_length": 16384,
  "datasets": [
    {"name": "source", "path": "/path/to/prefix", "weight": 1.0}
  ]
}
```

Each dataset path resolves to an indexed record prefix. A catalog may include
an immutable prefix inventory with record counts, file sizes, and `.idx`
checksums. Sources are sampled with replacement according to `weight`; records
within each source are sampled uniformly.

Build a catalog from a weighted `train_data_path` YAML:

```bash
PYTHONPATH=src python scripts/data/build_megatron_packed_config.py \
  --source-yaml /path/to/train.yaml \
  --output /path/to/packed-data.json \
  --max-open-files 64 \
  --prefix-block-size 256
```

Validate all physical prefixes before training:

```bash
PYTHONPATH=src python scripts/data/validate_megatron_packed.py \
  --config /path/to/packed-data.json
```

The packed collator applies the `sosp_eosp` supervision mask using
`<|extra_100|>` and `<|extra_101|>`. Token and mask parity can be checked with:

```bash
python scripts/data/validate_megatron_packed.py \
  --config /path/to/packed-data.json
```

Sequence training uses the same record format with repeated text and image
segments in a fixed 32K window. Set both `sequence_length` in the data catalog
and `data.max_length` in the training preset to `32768`.

## WebDataset

The generic WebDataset backend remains available for user-provided U0 IBQ
data. No project-specific WebDataset catalog is bundled.

Simple pair:

```text
sample.txt  # input text
sample.pt   # [H, W] integer target visual-token grid
```

Manifest sample:

```text
sample.json
sample.in.000.pt
sample.out.000.pt
sample.out.001.pt
```

The manifest contains ordered `input_segments` and a non-empty
`output_sequence`. Text and condition images are accepted as inputs; every
output image is supervised in manifest order.

```json
{
  "type": "webdataset",
  "format": "tokenized_tar",
  "seed": 42,
  "shuffle": true,
  "weighted_sampling": true,
  "samples_per_epoch": 100000,
  "skip_bad_samples": false,
  "datasets": [
    {
      "name": "dataset_a",
      "paths": ["/data/dataset_a/*.tar"],
      "visual_id_space": "raw",
      "weight": 1.0,
      "sample_count": 50000,
      "expected_shard_count": 50
    }
  ]
}
```

`visual_id_space=raw` means IDs are in `[0, visual_vocab_size)` and the
collator adds the model's visual-token offset. `visual_id_space=hf` means the
stored IDs already belong to the complete language-model vocabulary.

`weight` controls expected sample exposure, not loss weighting.
`sample_count` must describe the real dataset size, and
`expected_shard_count` prevents an incomplete glob from silently training.
Production configurations should keep `skip_bad_samples=false`.

Validate a user-provided catalog with:

```bash
python scripts/data/validate_webdataset.py \
  --config /path/to/webdataset.json --samples 32
```

Model weights, tokenizer weights, raw data, and tokenized shards are not part
of this repository.
