# Notice, licenses and credits

Hamsa Beta
Copyright 2026 Mostafa Drazi
Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

Hamsa is not affiliated with, endorsed by, or sponsored by any of the projects or companies named below. Names are used only to credit the work Hamsa builds on or was inspired by.

This file lists what ships in this repository, what Hamsa downloads at run time, and the terms that come with each. It is a good-faith summary, not legal advice; the linked licenses are the authority.

## 1. Bundled with the source

| Asset | Where | License |
| --- | --- | --- |
| Readex Pro typeface (Latin, Latin Extended, Arabic subsets), Copyright 2019 The Readex Pro Project Authors | `beta/assets/fonts/` (license text: `beta/assets/fonts/OFL.txt`) | SIL Open Font License 1.1. Free to use, embed and redistribute with the license text; the font may not be sold on its own. |
| Hamsa app icon and wordmark | `beta/assets/app-icon.*`, `beta/assets/hamsa-wordmark.svg` | Part of Hamsa, Apache-2.0 |
| UI icons | drawn in the app's own Slint code (`beta/ui/theme.slint`, `beta/ui/widgets.slint`) | Part of Hamsa, Apache-2.0 |

No other fonts, images, audio or model files are committed. Speech and language models are never part of the repository or the installer.

## 2. Downloaded by the app at run time (not redistributed here)

Hamsa downloads these on request, from the URLs pinned in `core/src/models.rs`, `core/src/llm/asr.rs`, `core/src/llm/cleanup.rs` and `core/src/llm/mod.rs`, and checks their SHA-256. They live on the user's disk only.

| Component | Made by | License | Notes |
| --- | --- | --- | --- |
| Phonon-2.5 and Phonon-2.5 English speech models (`FermionResearch/Phonon-2.5`, `FermionResearch/Phonon-2.5-English`) | Fermion Research | CC-BY-4.0 (Hugging Face model cards, checked 2026-11) | "Based on parakeet-tdt-0.6b-v3 by NVIDIA". Derived from NVIDIA Parakeet-TDT 0.6B v3. The model cards say the model code and the command-line tool in their repositories are Apache-2.0. The cards reference a `NOTICE` file listing the changes made to the original. |
| Parakeet-TDT 0.6B v3 (the base of Phonon) | NVIDIA | CC-BY-4.0 (`nvidia/parakeet-tdt-0.6b-v3`) | CC-BY-4.0 requires credit to the creator, a link to the license, and a note of changes. The Hamsa Beta Help page, this file and the README carry that credit. |
| fihrIA-ASR 0.6B and 1.7B, GGUF builds (`fihriaLab/fihrIA-ASR-0.6B-GGUF`, `fihriaLab/fihrIA-ASR-1.7B-GGUF`), Moroccan Darija speech recognition | fihrIA lab (fihriaLab) | Apache-2.0 (model cards) | Fine-tunes of Qwen3-ASR (Qwen/Qwen3-ASR-0.6B and 1.7B, Qwen team, Alibaba Cloud). The cards ask for no extra attribution beyond a citation block: "fihrIA-ASR: Open-Weight Speech Recognition and Semantic Evaluation for Moroccan Darija", fihrIA, 2026. |
| SpeakoFlow Mini 0.8B, GGUF builds (`SpeakoFlow/speakoflow-mini`), optional Smart cleanup | SpeakoFlow (the card credits Abhishek Barali) | Apache-2.0 (model card) | Fine-tuned from Qwen3.5-0.8B (Apache-2.0) with LoRA, per the card. |
| llama.cpp release `b12651`, Windows Vulkan x64 build (`llama-server`) | ggml-org and contributors | MIT | Started as a separate local process on 127.0.0.1 only while a Darija model or Smart cleanup is in use. |

The model cards were read on their Hugging Face pages. Model licenses can change; re-check before each release.

## 3. Slint (user interface toolkit)

Hamsa Beta's windows are built with [Slint](https://slint.dev) 1.19 (`slint`, `i-slint-core` and related crates). Slint is offered under three licenses: GPL-3.0-only, the Slint Royalty-free Desktop, Mobile and Web Applications License 2.0, or a paid commercial license.

Hamsa Beta uses Slint under the **Royalty-free License 2.0** (text: `LICENSES/LicenseRef-Slint-Royalty-free-2.0.md` inside the `slint` crate; summary at <https://slint.dev/terms-and-conditions>). Its conditions for a desktop application:

1. Attribution, **either**
   - (a) show the `AboutSlint` widget in an "About" screen or dialog reachable from the application's top-level menu (or on the splash screen if there is no such screen), **or**
   - (b) show the [Slint attribution badge](https://github.com/slint-ui/slint/tree/master/logo/MadeWithSlint-logo-whitebg.png) on a public web page, preferably where the application's binaries are downloaded, where visitors can easily find it.
2. No use in embedded systems; no distribution of Slint alone; no distribution of an application that exposes Slint's APIs; keep license notices in Slint's source intact.

How Hamsa meets it: the Help page of the app shows `AboutSlint` (`beta/ui/main.slint`, Credits section), and the README, which is where the releases are linked, shows the "Made with Slint" badge.

If you fork Hamsa, check that you can meet these terms too, or use Slint under GPL-3.0 (in which case the combined work you distribute must itself be GPL-3.0 compatible) or the commercial license.

## 4. Other design and algorithm credits

- NVIDIA NeMo's Parakeet-TDT: the transducer decoding loop that Phonon re-implements.
- `eschmidbauer/phonon-2.5-c` (MIT): the reference for the Phonon container layout and decoding loop (see `phonon/src/lib.rs`).
- Hugging Face `transformers`' `ParakeetForTDT`: the reference for the computation.

## 5. Rust dependencies

Resolved with `cargo metadata` for `x86_64-pc-windows-msvc` (409 packages, direct and transitive). Only Slint is not under a plain permissive license, and it is dual/triple licensed as described above.

| License (SPDX, as declared by the crates) | Crates |
| --- | --- |
| MIT OR Apache-2.0 (also written Apache-2.0 OR MIT, MIT/Apache-2.0) | 247 |
| MIT | 55 |
| Apache-2.0 | 16 |
| Unicode-3.0 (ICU data crates) | 27 |
| GPL-3.0-only OR Slint Royalty-free-2.0 OR Slint Software-3.0 | 12 (the Slint crates) |
| BSD-3-Clause, BSD-2-Clause and combinations | about 15 |
| Zlib, ISC, BSL-1.0, CC0-1.0, Unlicense, 0BSD, CDLA-Permissive-2.0 and combinations | the rest |

Main direct dependencies:

| Crate | License | Used for |
| --- | --- | --- |
| `slint`, `slint-build` 1.19 | see section 3 | UI |
| `wgpu` 22 | MIT OR Apache-2.0 | GPU compute in Phonon |
| `winit` 0.30 (via Slint) | Apache-2.0 | windowing |
| `cpal` 0.15 | Apache-2.0 | microphone capture |
| `rubato` 0.15 | MIT | resampling to 16 kHz |
| `hound` 3.5 | Apache-2.0 | WAV files |
| `rayon`, `realfft`, `matrixmultiply`, `half`, `bytemuck`, `pollster` | MIT / Apache-2.0 | Phonon numerics |
| `zstd` 0.13 | BSD-3-Clause | unpacking model archives |
| `tar`, `zip`, `sha2`, `ureq` | MIT OR Apache-2.0 / MIT | model downloads and checks |
| `arboard` 3, `tray-icon` 0.26, `ttf-parser` 0.25 | MIT OR Apache-2.0 | clipboard, tray, font metrics |
| `serde`, `serde_json` | MIT OR Apache-2.0 | settings files |
| `windows-sys` 0.59 | MIT OR Apache-2.0 | Win32 calls |
| `embed-resource` 3 | MIT | embeds the Windows icon |

To regenerate the full list: `cargo metadata --format-version 1 --filter-platform x86_64-pc-windows-msvc`.

## 6. Inspiration

Hamsa is an independent project, inspired by the idea behind Wispr Flow and by Google's on-device dictation work (Google AI Edge Eloquent, Gboard's Rambler). No code, assets or branding from these products are used, and Hamsa is not affiliated with or endorsed by their makers.
