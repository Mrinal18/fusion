# fusion

### Package Architecture

- 11/08/2020: Drafted the architecture of the models using [PlantUML](https://plantuml.com/)
- 11/10/2020: Made project as a python package and created the directory structure.


### Coding ToDo List:
- architecture
  - DCGAN
    - VaeEncoder
    - VaeDecoder
- model
  - MMVAE
- criterion
  - mi_estimator
    - critic
      - Bilinear
    - Donsker Varadhan Loss
  - misc
    - cca
    - mmvae
- task
  - linear evaluation
  - logreg evaluation
  - saliency
- configs
  - needed yaml with hydra for all models
