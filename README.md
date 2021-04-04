# fusion

### Package Architecture

The architecture of the models using [PlantUML](https://plantuml.com/)

### Coding ToDo List:
- architecture
  - DCGAN
    - VaeEncoder
    - VaeDecoder
- model
  - MMVAE
  - S-AE
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
  - logreg evaluation
  - saliency
- configs
  - hydra configs for all models
