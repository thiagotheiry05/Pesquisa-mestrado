Esta pasta contém a implementação completa da metodologia proposta na dissertação **"Mapeamento automatizado da fronteira entre as crostas continental e oceânica (COB/COT) baseado em Aprendizado de Máquina"**, desenvolvida no Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPgEEC/UFRN).

Nesta versão da metodologia foram implementadas e avaliadas quatro arquiteturas de aprendizado profundo para classificação de *patches* sísmicos e posterior reconstrução da segmentação da imagem original.

## Estrutura

A pasta está organizada da seguinte forma:

```
Metodologia Proposta/
│
├── CNN Baseline/
│   ├── notebook.ipynb
│   └── resultados/
│
├── CNN com Informação Posicional/
│   ├── notebook.ipynb
│   └── resultados/
│
├── Transformer/
│   ├── notebook.ipynb
│   └── resultados/
│
└── CNN-Transformer/
    ├── notebook.ipynb
    └── resultados/
```

Cada diretório contém:

- Notebook com toda a implementação da arquitetura;
- Processo de treinamento;
- Etapa de inferência;
- Reconstrução da imagem segmentada;
- Geração das métricas de avaliação;
- Resultados obtidos durante os experimentos.

## Sobre o conjunto de dados

O conjunto de dados utilizado nesta pesquisa foi construído a partir de imagens sísmicas interpretadas por geofisicos, sendo composto por *patches* previamente rotulados em cinco classes geológicas.

Por questões relacionadas à disponibilidade e aos direitos de uso dos dados, o dataset utilizado nesta dissertação **não está disponível publicamente neste repositório**.

Pesquisadores interessados em reproduzir integralmente os experimentos deverão solicitar autorização para utilização do conjunto de dados junto aos responsáveis pelo projeto ou à instituição detentora dos dados.

## Reprodução dos experimentos

Após a obtenção do conjunto de dados, cada arquitetura pode ser executada individualmente por meio do respectivo notebook.

Os notebooks foram organizados de forma sequencial, contemplando todas as etapas do experimento:

1. Carregamento do conjunto de dados;
2. Pré-processamento;
3. Treinamento do modelo;
4. Avaliação quantitativa;
5. Geração das matrizes de confusão;
6. Geração dos mapas de confiança (*Softmax*);
7. Reconstrução da segmentação da imagem sísmica.

## Observações

Este repositório foi disponibilizado com o objetivo de favorecer a transparência e a reprodutibilidade da metodologia proposta nesta dissertação.

Caso seja autorizado o acesso ao conjunto de dados, os notebooks permitem reproduzir integralmente o pipeline experimental descrito no Capítulo 4 da dissertação.