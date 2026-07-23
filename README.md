# Mapeamento Automatizado da Fronteira Continental-Oceânica (COB/COT)

Este repositório reúne os códigos desenvolvidos durante a pesquisa de mestrado intitulada:

> **Mapeamento automatizado da fronteira entre as crostas continental e oceânica utilizando arquiteturas de Aprendizado de Máquina**

O objetivo da pesquisa foi avaliar diferentes arquiteturas de aprendizado profundo para a classificação de *patches* sísmicos e a segmentação automática de seções sísmicas, comparando abordagens baseadas em Redes Neurais Convolucionais (CNNs), Transformers e arquiteturas híbridas.

---

## Organização do repositório

O projeto está dividido em duas partes principais:

```
.
├── Explorações Iniciais/
└── Metodologia Proposta/
```

### Explorações Iniciais

Esta pasta contém os primeiros experimentos realizados durante o desenvolvimento da pesquisa.

As implementações presentes nesta etapa correspondem a abordagens investigadas antes da definição da metodologia final descrita na dissertação, incluindo estratégias para geração do conjunto de dados e diferentes formas de criação dos *patches* sísmicos.

Esses experimentos foram mantidos no repositório por representarem a evolução metodológica da pesquisa.

---

### Metodologia Proposta

Esta pasta contém a implementação final utilizada na dissertação.

Nela encontram-se as quatro arquiteturas avaliadas experimentalmente:

- CNN Baseline;
- CNN com Informação Posicional;
- Transformer;
- CNN–Transformer.

Cada arquitetura possui seu próprio notebook, contendo todas as etapas do experimento, além de uma pasta com os respectivos resultados obtidos durante as avaliações.

---

## Conjunto de dados

O conjunto de dados utilizado nesta pesquisa foi construído a partir de interpretações geológicas de imagens sísmicas.

Por questões relacionadas aos direitos de uso e à disponibilidade dos dados, o dataset **não está disponibilizado publicamente neste repositório**.

Pesquisadores interessados na reprodução dos experimentos deverão solicitar autorização de acesso aos responsáveis pelo conjunto de dados.

---

## Reprodutibilidade

Todo o código desenvolvido para a implementação da metodologia proposta encontra-se disponível neste repositório.

Após a obtenção do conjunto de dados e a configuração do ambiente de execução, os notebooks permitem reproduzir o pipeline experimental descrito na dissertação, incluindo:

- preparação dos dados;
- treinamento dos modelos;
- avaliação quantitativa;
- geração das matrizes de confusão;
- geração dos mapas de confiança (*Softmax*);
- reconstrução das imagens segmentadas.

As instruções específicas de execução encontram-se descritas no arquivo **README.md** da pasta **Metodologia Proposta**.

---

## Autor

**Thiago Theiry de Oliveira**

Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPgEEC)  
Universidade Federal do Rio Grande do Norte (UFRN)

---

## Licença

Este repositório foi disponibilizado para fins acadêmicos e científicos.

Caso utilize este trabalho em pesquisas ou publicações, recomenda-se citar a dissertação correspondente.
