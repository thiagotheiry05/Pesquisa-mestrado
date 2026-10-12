
# Mapeamento Automatizado da Fronteira Continental-Oceânica (COB/COT)

Este repositório reúne os códigos desenvolvidos durante a pesquisa de mestrado intitulada:

**Mapeamento automatizado da fronteira entre as crostas continental e oceânica utilizando arquiteturas de Aprendizado de Máquina**

O objetivo da pesquisa foi avaliar diferentes arquiteturas de aprendizado profundo para a classificação de *patches* sísmicos e a segmentação automática de seções sísmicas, comparando abordagens baseadas em Redes Neurais Convolucionais (CNNs), Transformers e arquiteturas híbridas.

---

## Organização do repositório

### Metodologia proposta

A pasta [`metodologia-proposta/`](metodologia-proposta/) contém a implementação da metodologia híbrida baseada em CNN e Transformer, com utilização de informações posicionais.

Nessa pasta estão disponíveis:

- Código-fonte da metodologia proposta.
- Configurações dos experimentos.
- Procedimentos de pré-processamento dos dados sísmicos.
- Implementação da arquitetura CNN + Transformer.
- Rotinas de treinamento e avaliação.
- Reconstrução das seções sísmicas segmentadas.
- Geração de mapas de probabilidade Softmax.
- Instruções detalhadas para instalação, configuração e execução do projeto.

### Guia de instalação e execução

Para reproduzir os experimentos ou utilizar o modelo em outras seções sísmicas, consulte o manual disponível na pasta da metodologia proposta.

**[Acessar o guia completo de instalação e execução](metodologia-proposta/README.md)**
