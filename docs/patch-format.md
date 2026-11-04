# Patch format (`weightsmith.patch/v0`)

A patch records the exact difference between two weight sets with the same parameter names, shapes and dtypes. Applying it reproduces the result byte for byte. Applying it in reverse reproduces the original.

## Layout

```
patch/
  manifest.json        description, hashes, per-tensor entries, provenance
  patch.safetensors    tensor payloads (absent when nothing changed)
```

## Change semantics

A parameter is changed when its stored bits differ. That is stricter than numeric equality: `-0.0` and `0.0` differ, and two NaNs with identical bits do not. Bitwise comparison is what makes the round trip exact and content hashes reproducible.

## Encoding

For each changed tensor `name`, in sorted order:

| Encoding | When | Payload keys |
|---|---|---|
| `sparse` | changed entries are fewer than 25% of the tensor | `name::indices` (int64 flat indices, ascending), `name::old`, `name::new` (values at those indices) |
| `dense` | otherwise | `name::old`, `name::new` (whole flattened tensors) |

Values are stored in the tensor's own dtype. Version 0 stores replacement values, not deltas. Floating-point deltas would not round-trip exactly.

## Manifest

```json
{
  "format": "weightsmith.patch/v0",
  "base_model_id": "sha256:...",
  "result_model_id": "sha256:...",
  "changed_params": 130,
  "total_params": 4996,
  "entries": [
    {"name": "layers.1.bias", "shape": [64], "dtype": "F32", "encoding": "sparse",
     "changed": 2, "numel": 64}
  ],
  "provenance": {"weightsmith": "0.1.0", "program": "RepairMLP", "program_hash": "...", "...": "..."}
}
```

## Content hash

`sha256:` plus the hex SHA-256 over every tensor in name order. For each tensor the hash takes the UTF-8 name, a NUL byte, the dtype name, a NUL byte, the comma-separated shape, a NUL byte, and the tensor's raw little-endian bytes in row-major order.

## Applying

1. Hash the input. It must equal `base_model_id`, or `result_model_id` when reversing. Otherwise refuse.
2. For each entry, check the shape. For sparse entries, check that the input holds the expected `old` values (`new` when reversing) at the indices. Then write `new` (`old` when reversing).
3. Hash the output. It must equal `result_model_id` (`base_model_id` when reversing). Otherwise refuse.
