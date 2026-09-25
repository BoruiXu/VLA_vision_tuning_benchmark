# License scope and third-party notices

Newly authored VLA-Retain benchmark code and associated documentation are licensed under the [MIT License](LICENSE). The copyright notice uses anonymous contributor attribution for double-blind review.

Third-party components and upstream-derived code retain their original terms; the repository's MIT license does not replace those terms. Preserved license texts are in `licenses/`; pinned public sources are in `config/upstream.json`. In particular, `code/oft_finetune_base.py` derives from OpenVLA-OFT, `code/train_pi05_lora_pytorch.py` derives from OpenPI, and the patches/adapters extend their named upstream projects. Public upstream attribution must remain intact.

Model weights, training corpora and simulator assets are not redistributed. Their respective model/data/asset licenses apply independently. NORA's pinned source did not contain a top-level license file in the inspected tree; consult the upstream project before redistributing it. The setup script clones it only on explicit request.
