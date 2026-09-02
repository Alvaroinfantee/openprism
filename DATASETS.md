# Dataset acquisition and licensing

OpenPRISM does not redistribute third-party imagery. Download each archive from
its publisher, review its current terms, and stage it beneath `data/` using the
layout consumed by `openprism.datasets.DatasetCatalog`. Exact source revisions,
archive digests, and the locally validated inventory are recorded in
[`dataset_sources.json`](dataset_sources.json) and
[`dataset_inventory.json`](dataset_inventory.json).

| Dataset | Intended adapter use | Redistribution / use note |
|---|---|---|
| LLVIP | Low-light aligned RGB/infrared pairs and person boxes | Publisher terms are non-commercial; verify the current dataset page before use. |
| MSRS | Day/night road scenes, semantic masks, and a detection subset | The upstream repository did not state an explicit dataset license when this release was prepared; obtain clarification before redistribution or commercial use. |
| Caltech Aerial RGB-T | Aerial RGB, 8/16-bit thermal evidence, and terrain semantics | The publisher states non-commercial research use; preserve its terms and citation. |

## Official acquisition locations

- **LLVIP:** follow the publisher's
  [download instructions](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)
  and accept its [custom non-commercial terms](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md).
- **MSRS:** obtain the files from the
  [upstream repository](https://github.com/Linfeng-Tang/MSRS). The repository
  had no explicit license when this manifest was prepared, so OpenPRISM does
  not mirror it.
- **Caltech Aerial RGB-T:** download version 1.0.0 from the
  [publisher record](https://doi.org/10.22002/cks6g-ps927) under CC BY-NC 4.0.

The three extracted datasets occupy approximately 8.8 GiB; retained source
archives occupy another 8.35 GiB. GitHub is not the authoritative data host,
and duplicating these archives would obscure publisher terms and provenance.

Expected top-level directories are `data/LLVIP`, `data/MSRS`, and
`data/Caltech_Aerial_RGBT`. Run:

```bash
python tooling/validate_multispectral_datasets.py
```

The validator checks pairing and expected inventory without rewriting publisher
files. Dataset samples, GPS traces, and captured subjects can carry privacy or
location risk even when a download is permitted. Follow institutional review,
data protection, and aviation requirements applicable to your work.
