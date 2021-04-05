# Fusion
Fusion is a self-supervised framework for data with multiple sources — specifically, this framework aims to support neuroimaging applications.

Fusion aims to provide a foundation for fair comparison of new models in different multi-view, multi-domain, or multi-modal scenarios. Currently, we provide only two datasets with multi-view, multi-domain natural images. Further, the repository will be updated with multi-modal neuroimaging datasets.

The other goal of Fusion is to reproduce the following works:
https://arxiv.org/abs/2012.13623, https://arxiv.org/abs/2012.13619, and https://arxiv.org/abs/2103.15914.

This project is under active development, and the codebase is subject to change.

---
## Installation
To install requirements:
```
pip install requirements.txt
```
To install in standard mode:
```
pip install .
```
To install in development mode:
```
pip install -e .
```
---
## Experiments
To run a default experiment, use:
```
python main.py
```
The default experiment will train the XX model on the Two-View MNIST dataset.

The code is written mainly with PyTorch (https://pytorch.org/).

The experiments are defined using the Hydra configs (https://hydra.cc/docs/next/intro) and located in the directory `configs`.

The training pipeline is based on the Catalyst framework (https://catalyst-team.github.io/catalyst/).

---
## Questions
If you have any questions about implementation and training, don't hesitate to either open an issue here or send an email to eidos92@gmail.com.

---
## Development

### Roadmap:
Last updated April 4th, 2021

Current ToDo list:
- Documentation
- Typing
- Code
  - architecture
    - DCGAN
      - VaeEncoder
      - VaeDecoder
  - dataset
    - Dataset class for OASIS3
  - model
    - MMVAE
  - criterion
    - loss
      - S-AE
      - L-CCA
      - DCCAE
    - mi_estimator
      - critic
        - Bilinear
      - Donsker Varadhan Loss
    - misc
      - cca
      - mmvae
  - task
    - logreg evaluation with optuna
    - saliency
    - out of distribution generalization
  - configs
    - hydra configs for all models
---
### Package Architecture

The package's software architecture is available in the directory `UML` based on [PlantUML](https://plantuml.com/).

---
## Citation

If you use Fusion for published work, please cite our work using the following bibtex entry.

For taxonomy and natural images please cite:

```
@misc{fedorov2021selfsupervised,
      title={Self-Supervised Multimodal Domino: in Search of Biomarkers for Alzheimer's Disease},
      author={Alex Fedorov and Tristan Sylvain and Eloy Geenjaar and Margaux Luck and Lei Wu and Thomas P. DeRamus and Alex Kirilin and Dmitry Bleklov and Vince D. Calhoun and Sergey M. Plis},
      year={2021},
      eprint={2012.13623},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

For Neuroimaging please cite:
```
@article{fedorov2020self,
  title={On self-supervised multi-modal representation learning: An application to Alzheimer's disease},
  author={Fedorov, Alex and Wu, Lei and Sylvain, Tristan and Luck, Margaux and DeRamus, Thomas P and Bleklov, Dmitry and Plis, Sergey M and Calhoun, Vince D},
  journal={arXiv preprint arXiv:2012.13619},
  year={2020}
}
```

For out-of-distribution generalization please cite:
```
@misc{fedorov2021tasting,
      title={Tasting the cake: evaluating self-supervised generalization on out-of-distribution multimodal MRI data},
      author={Alex Fedorov and Eloy Geenjaar and Lei Wu and Thomas P. DeRamus and Vince D. Calhoun and Sergey M. Plis},
      year={2021},
      eprint={2103.15914},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
---
## Acknowledgement
This work is supported by NIH R01 EB006841.

Data were provided in part by OASIS-3: Principal
Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P50
AG00561, P30 NS09857781, P01 AG026276, P01 AG003991,
R01 AG043434, UL1 TR000448, R01 EB009352. AV-45
doses were provided by Avid Radiopharmaceuticals, a
wholly-owned subsidiary of Eli Lilly.
