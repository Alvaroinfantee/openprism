# Dataset acquisition and licensing

OpenPRISM does not redistribute third-party imagery. Download each archive from
its publisher, review its current terms, and stage it beneath `data/` using the
layout consumed by `openprism.datasets.DatasetCatalog`.

| Dataset | Intended adapter use | Redistribution / use note |
|---|---|---|
| LLVIP | Low-light aligned RGB/infrared pairs and person boxes | Publisher terms are non-commercial; verify the current dataset page before use. |
| MSRS | Day/night road scenes, semantic masks, and a detection subset | The upstream repository did not state an explicit dataset license when this release was prepared; obtain clarification before redistribution or commercial use. |
| Caltech Aerial RGB-T | Aerial RGB, 8/16-bit thermal evidence, and terrain semantics | The publisher states non-commercial research use; preserve its terms and citation. |

Expected top-level directories are `data/LLVIP`, `data/MSRS`, and
`data/Caltech_Aerial_RGBT`. Run:

```bash
python tooling/validate_multispectral_datasets.py
```

The validator checks pairing and expected inventory without rewriting publisher
files. Dataset samples, GPS traces, and captured subjects can carry privacy or
location risk even when a download is permitted. Follow institutional review,
data protection, and aviation requirements applicable to your work.
