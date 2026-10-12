
# Classificação de Estruturas Geológicas em Imagens Sísmicas com CNN e Transformer

Este repositório contém a implementação de uma metodologia de classificação de estruturas geológicas em imagens sísmicas, utilizando uma arquitetura híbrida baseada em **Redes Neurais Convolucionais (CNN)**, **Transformer** e **informações posicionais**.

O projeto foi desenvolvido no contexto de uma pesquisa acadêmica e disponibilizado para permitir a reprodução dos experimentos e a aplicação do modelo em outras seções sísmicas.

O principal objetivo deste repositório é fornecer um ambiente organizado para treinamento, avaliação e inferência, facilitando sua utilização por pesquisadores das áreas de Geologia, Geofísica e Computação.

---

## 1. Visão geral da metodologia

A metodologia divide uma seção sísmica em pequenas regiões, chamadas *patches*, e utiliza um modelo de aprendizado profundo para classificar cada região de acordo com a estrutura geológica predominante.

A arquitetura combina três componentes:

- **CNN:** responsável pela extração de características das imagens sísmicas.
- **Transformer:** responsável pelo processamento das características extraídas, utilizando mecanismos de atenção.
- **Informações posicionais:** representam a localização normalizada de cada patch na seção sísmica.

Após a classificação, os patches são reorganizados em suas posições originais, permitindo reconstruir uma imagem segmentada da seção sísmica.

### Classes geológicas

O modelo considera cinco classes:

| Classe | Cor utilizada |
|---|---|
| Água | Azul |
| Bacia sedimentar | Amarelo |
| Crosta continental | Marrom |
| Crosta oceânica | Laranja |
| Manto | Cinza |

Cada patch recebe uma classificação, representada pela cor correspondente.

---

## 2. Dados utilizados

O experimento utiliza uma seção sísmica e uma máscara de referência para treinamento e avaliação.

Os arquivos utilizados são:

**Seção sísmica original:**

`BB08RE11-210-PRCMIG.npy`

**Máscara de referência:**

`Mascara_210_4.png`

A imagem sísmica está armazenada no formato NumPy (`.npy`), enquanto a máscara contém as regiões geológicas identificadas por cores.

Também é utilizada uma segunda seção sísmica para aplicação do modelo em dados não utilizados durante o treinamento:

`tgs-grc2001_line01a-12_fin-mgr_tapeXfile15.npy`

Os arquivos de dados não estão incluídos neste repositório. Eles precisam ser obtidos separadamente e adicionados às pastas indicadas no guia de execução.

---

## 3. Pré-processamento

A seção sísmica é dividida em patches de **64 × 64 pixels**.

Durante o pré-processamento:

1. A imagem e a máscara são carregadas.
2. A seção é dividida em patches.
3. Cada patch recebe uma classe de acordo com a cor predominante na máscara.
4. São calculadas as coordenadas normalizadas de cada patch.
5. Os dados são divididos em conjuntos de treinamento, validação e teste.

A divisão utilizada no experimento é:

| Conjunto | Proporção |
|---|---|
| Treinamento | 70% |
| Validação | 15% |
| Teste | 15% |

O conjunto de treinamento também passa por técnicas de aumento de dados (*data augmentation*), utilizadas para ampliar a diversidade das amostras e balancear as classes.

**Observação:** a divisão é realizada entre patches da seção original. Portanto, a avaliação nesse conjunto de teste não equivale a uma avaliação independente em uma seção sísmica completamente diferente.

---

## 4. Treinamento e avaliação

O treinamento utiliza a arquitetura CNN + Transformer implementada em TensorFlow/Keras.

As principais configurações estão disponíveis no arquivo:

`src/metodologia_proposta/config.py`

Entre os parâmetros configuráveis estão:

- Quantidade máxima de épocas.
- Tamanho do batch.
- Tamanho dos patches.
- Parâmetros do Transformer.
- Quantidade de amostras utilizadas no augmentation.
- Critério de parada antecipada (*Early Stopping*).

Durante o treinamento, o melhor modelo é salvo automaticamente em:

`models/best_model.keras`

A avaliação utiliza métricas como:

- Acurácia.
- Precisão (*Precision*).
- Revocação (*Recall*).
- F1-score.
- Matriz de confusão.

Também são gerados gráficos de evolução da perda (*loss*) e da acurácia durante o treinamento.

---

## 5. Reconstrução e mapas Softmax

Após o treinamento, o modelo pode classificar todos os patches de uma seção sísmica e reconstruir sua representação geológica.

O projeto gera:

**Imagem segmentada:** reconstrução da seção utilizando as cores das classes previstas.

**Mapas Softmax:** imagens que representam a probabilidade atribuída pelo modelo a cada classe geológica.

**Mapa de confiança:** representação da maior probabilidade Softmax encontrada para cada patch.

**Figura Softmax combinada:** visualização dos cinco mapas de probabilidade em uma única imagem.

As probabilidades Softmax representam a confiança produzida pelo modelo, mas não garantem que a classificação esteja correta.

---

# 6. Guia de instalação e execução

Esta seção apresenta o procedimento necessário para reproduzir o experimento em um computador Windows.

Os comandos foram preparados para utilização no **PowerShell**, disponível no terminal do Visual Studio Code.

## 6.1. Programas necessários

Antes de executar o projeto, é necessário instalar:

| Programa | Finalidade |
|---|---|
| Python 3.12 | Linguagem utilizada no projeto |
| Git | Download e gerenciamento do repositório |
| Visual Studio Code | Editor de código |
| uv | Gerenciamento do ambiente Python e das dependências |

### Instalar Python

Acesse:

https://www.python.org/downloads/

Instale uma versão do Python 3.12.

Durante a instalação, caso a opção esteja disponível, marque:

`Add Python to PATH`

Depois, abra o PowerShell e execute:

```powershell
python --version
```

O resultado esperado é semelhante a:

```text
Python 3.12.10
```

Caso o comando `python` não seja reconhecido, verifique se o Python foi adicionado ao PATH.

### Instalar Git

Acesse:

https://git-scm.com/downloads

Instale o Git utilizando as configurações padrão.

Para verificar a instalação:

```powershell
git --version
```

### Instalar Visual Studio Code

Acesse:

https://code.visualstudio.com/

Instale o programa.

Recomenda-se instalar também a extensão **Python**, disponibilizada pela Microsoft, dentro do próprio VS Code.

### Instalar uv

O `uv` é utilizado para criar o ambiente Python e instalar automaticamente as bibliotecas necessárias.

No PowerShell, execute:

```powershell
winget install --id=astral-sh.uv -e
```

Depois, abra um novo terminal e verifique:

```powershell
uv --version
```

O projeto foi desenvolvido utilizando `uv` e possui um arquivo `uv.lock`, que registra as dependências utilizadas.

---

## 6.2. Baixar o projeto do GitHub

Abra o PowerShell e navegue até a pasta onde deseja armazenar o projeto.

Por exemplo:

```powershell
cd "$HOME\Documents"
```

Clone o repositório:

```powershell
git clone https://github.com/Pesquisa-mestrado/metodologia-proposta.git
```

Entre na pasta do projeto:

```powershell
cd metodologia-proposta
```

Para abrir o projeto no Visual Studio Code, execute:

```powershell
code .
```

Se o comando `code` não estiver disponível, abra o VS Code manualmente e selecione:

**File → Open Folder → metodologia-proposta**

---

## 6.3. Preparar o ambiente Python

Com o terminal aberto na raiz do projeto, execute:

```powershell
uv sync
```

Esse comando realiza automaticamente as seguintes operações:

1. Verifica a versão de Python necessária.
2. Cria o ambiente virtual `.venv`, caso ainda não exista.
3. Instala as bibliotecas definidas no projeto.
4. Instala o pacote Python do próprio projeto.
5. Utiliza o arquivo `uv.lock` para reproduzir as dependências.

Não é necessário instalar TensorFlow, NumPy, OpenCV ou as demais bibliotecas individualmente.

Para verificar se o ambiente está funcionando:

```powershell
uv run python --version
```

Também é possível verificar o TensorFlow:

```powershell
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

Se a versão do TensorFlow aparecer, a biblioteca foi carregada corretamente.

**Importante:** o TensorFlow utilizado no ambiente Windows pode executar pela CPU, mesmo que o computador possua uma placa de vídeo NVIDIA. Isso não impede a execução, mas o treinamento pode demorar mais.

---

## 6.4. Organização das pastas

A estrutura principal do projeto é:

```text
metodologia-proposta/
|
|-- data/
|   |-- raw/
|   |-- unseen/
|
|-- models/
|
|-- outputs/
|   |-- figures/
|       |-- original/
|       |-- unseen/
|
|-- src/
|   |-- metodologia_proposta/
|       |-- augmentation.py
|       |-- config.py
|       |-- data.py
|       |-- evaluation.py
|       |-- inference.py
|       |-- model.py
|       |-- patches.py
|       |-- preprocessing.py
|       |-- reconstruction.py
|       |-- softmax.py
|       |-- train.py
|       |-- visualization.py
|
|-- main.py
|-- pyproject.toml
|-- uv.lock
|-- README.md
```

As principais pastas possuem as seguintes finalidades:

| Pasta | Finalidade |
|---|---|
| `data/raw/` | Dados sísmicos e máscara utilizados no treinamento |
| `data/unseen/` | Seção sísmica não utilizada no treinamento |
| `models/` | Armazenamento do modelo treinado |
| `outputs/` | Resultados e imagens gerados |
| `src/metodologia_proposta/` | Código-fonte da metodologia |

---

## 6.5. Adicionar os dados sísmicos

Os arquivos sísmicos precisam ser adicionados manualmente ao projeto.

### Dados de treinamento

Coloque os arquivos:

```text
BB08RE11-210-PRCMIG.npy
Mascara_210_4.png
```

Na pasta:

```text
data/raw/
```

A estrutura deverá ficar assim:

```text
data/
|-- raw/
|   |-- BB08RE11-210-PRCMIG.npy
|   |-- Mascara_210_4.png
```

### Seção sísmica não vista

Coloque o arquivo:

```text
tgs-grc2001_line01a-12_fin-mgr_tapeXfile15.npy
```

Na pasta:

```text
data/unseen/
```

A estrutura final será:

```text
data/
|-- raw/
|   |-- BB08RE11-210-PRCMIG.npy
|   |-- Mascara_210_4.png
|
|-- unseen/
|   |-- tgs-grc2001_line01a-12_fin-mgr_tapeXfile15.npy
```

**Atenção:** os nomes dos arquivos devem corresponder aos nomes configurados no código. Caso os arquivos possuam nomes diferentes, será necessário ajustar os respectivos caminhos.

---

## 6.6. Configurar o experimento

As configurações principais estão no arquivo:

```text
src/metodologia_proposta/config.py
```

Abra esse arquivo no Visual Studio Code.

### Executar um novo treinamento

Para treinar um modelo desde o início, configure:

```python
TRAIN_MODEL = True
```

Defina a quantidade máxima de épocas:

```python
EPOCHS = 50
```

E o tamanho do batch:

```python
BATCH_SIZE = 32
```

O treinamento utilizará o conjunto de treinamento, executará a validação e salvará o melhor modelo encontrado.

O projeto utiliza parada antecipada (*Early Stopping*). Portanto, o treinamento poderá terminar antes de completar as 50 épocas.

### Utilizar um modelo já treinado

Caso já exista um modelo salvo, configure:

```python
TRAIN_MODEL = False
```

Nesse caso, o projeto carregará:

```text
models/best_model.keras
```

O arquivo precisa existir nessa pasta.

**Importante:** o repositório não inclui automaticamente os pesos do modelo treinado. Para executar sem treinamento, é necessário obter previamente um arquivo `best_model.keras` compatível ou realizar um treinamento.

Na versão atual, mesmo com `TRAIN_MODEL = False`, o programa ainda executa o carregamento e o processamento dos dados necessários à avaliação, além das reconstruções e inferências configuradas em `main.py`.

---

## 6.7. Executar o projeto

Após instalar as dependências, adicionar os dados e configurar o experimento, abra o terminal na raiz do projeto.

Execute:

```powershell
uv run python main.py
```

O programa executará as etapas configuradas no código.

### Etapa 1 — Carregamento dos dados

O programa carrega a seção sísmica e a máscara de referência.

### Etapa 2 — Geração dos patches

A imagem é dividida em patches de 64 × 64 pixels, com suas respectivas classes e posições.

### Etapa 3 — Preparação dos conjuntos

Os patches são separados em treinamento, validação e teste.

O programa também realiza o aumento de dados do conjunto de treinamento.

### Etapa 4 — Treinamento ou carregamento do modelo

Se `TRAIN_MODEL = True`, o modelo será treinado.

Se `TRAIN_MODEL = False`, o modelo salvo será carregado.

### Etapa 5 — Avaliação

O modelo é avaliado utilizando o conjunto de teste.

O terminal apresenta informações como:

```text
Test loss: ...
Test accuracy: ...

Relatório de classificação:
precision    recall    f1-score    support
...
```

Também é gerada a matriz de confusão.

### Etapa 6 — Reconstrução da seção original

O modelo classifica os patches da seção original e reconstrói uma imagem segmentada.

### Etapa 7 — Mapas Softmax

São gerados mapas de probabilidade para cada uma das cinco classes geológicas e um mapa de confiança.

### Etapa 8 — Inferência na segunda linha

O programa carrega a seção sísmica não utilizada no treinamento e realiza sua classificação.

Também são gerados os mapas Softmax dessa nova seção.

**Observação:** o tempo de execução depende do processador, da memória disponível, do tamanho das imagens e da configuração de treinamento.

---

## 6.8. Onde encontrar os resultados

Os resultados são armazenados automaticamente na pasta:

```text
outputs/
```

### Avaliação e reconstrução

Na pasta `outputs/figures/`, são gerados arquivos como:

| Arquivo | Conteúdo |
|---|---|
| `training_loss.png` | Evolução da perda durante o treinamento |
| `training_accuracy.png` | Evolução da acurácia durante o treinamento |
| `confusion_matrix.png` | Matriz de confusão |
| `segmented_image.png` | Reconstrução da seção original |
| `reference_mask.png` | Máscara de referência |

Os gráficos de treinamento são gerados quando o treinamento é executado.

### Mapas Softmax da seção original

Na pasta:

```text
outputs/figures/original/
```

São gerados:

```text
softmax_agua.png
softmax_bacia_sedimentar.png
softmax_crosta_continental.png
softmax_crosta_oceanica.png
softmax_manto.png
softmax_confidence.png
softmax_combined.png
```

O arquivo `softmax_combined.png` reúne os mapas das cinco classes em uma única figura.

### Resultados da seção não vista

Na pasta:

```text
outputs/figures/unseen/
```

São gerados:

```text
segmented_image.png
softmax_agua.png
softmax_bacia_sedimentar.png
softmax_crosta_continental.png
softmax_crosta_oceanica.png
softmax_manto.png
softmax_confidence.png
softmax_combined.png
```

Esses arquivos representam a classificação e as probabilidades produzidas pelo modelo para a segunda seção sísmica.

---

## 6.9. Utilizar outra seção sísmica

É possível aplicar o modelo treinado a outra seção sísmica armazenada no formato `.npy`.

Para isso:

1. Coloque o novo arquivo `.npy` na pasta `data/unseen/`.
2. Abra `src/metodologia_proposta/config.py`.
3. Localize a configuração `UNSEEN_IMAGE_PATH`.
4. Altere o nome do arquivo para corresponder à nova seção.
5. Configure `TRAIN_MODEL = False`.
6. Certifique-se de que `models/best_model.keras` existe.
7. Execute novamente o projeto.

Exemplo:

```python
UNSEEN_IMAGE_PATH = (
    UNSEEN_DATA_DIR / "nova_linha_sismica.npy"
)
```

Em seguida:

```powershell
uv run python main.py
```

Os resultados serão gerados em `outputs/figures/unseen/`.

**Atenção:** arquivos gerados com os mesmos nomes poderão substituir resultados anteriores. Recomenda-se copiar os resultados para outra pasta antes de processar uma nova seção.

A nova seção deve ser compatível com a entrada esperada pelo modelo, incluindo a representação das amplitudes sísmicas. Diferenças entre levantamentos, escalas ou distribuições dos dados podem afetar as predições.

---

## 7. Problemas comuns

### Erro: `uv` não é reconhecido

Verifique se o `uv` foi instalado corretamente:

```powershell
uv --version
```

Se necessário, feche e abra novamente o terminal após a instalação.

### Erro: arquivo `.npy` não encontrado

Verifique se o arquivo está na pasta correta e se seu nome corresponde ao caminho configurado.

Exemplo:

```text
data/raw/BB08RE11-210-PRCMIG.npy
```

### Erro: modelo não encontrado

Se estiver utilizando:

```python
TRAIN_MODEL = False
```

Verifique se existe:

```text
models/best_model.keras
```

Caso não exista, será necessário treinar um modelo ou adicionar um modelo compatível à pasta.

### O treinamento está muito lento

O treinamento utiliza operações de aprendizado profundo e pode exigir bastante processamento.

Em ambientes Windows com TensorFlow executando pela CPU, o treinamento pode demorar consideravelmente.

Para testes iniciais, é possível utilizar:

```python
EPOCHS = 1
```

Entretanto, esse valor deve ser utilizado apenas para verificar o funcionamento do pipeline, não como configuração definitiva do experimento.

### Erro de memória

Imagens sísmicas grandes podem exigir bastante memória RAM, principalmente durante a reconstrução e a geração dos mapas Softmax.

Caso ocorra um erro de memória, recomenda-se fechar outros programas e verificar os recursos disponíveis.

Também poderá ser necessário adaptar o processamento para utilizar menos memória.

### Avisos do TensorFlow

Algumas mensagens exibidas pelo TensorFlow são apenas avisos relacionados ao ambiente de execução, otimizações ou disponibilidade de GPU.

Esses avisos não significam necessariamente que o programa falhou.

---

## 8. Observações para reprodução

Para reproduzir o experimento, recomenda-se manter os parâmetros definidos em `config.py` e utilizar os arquivos de dados correspondentes.

O arquivo `uv.lock` deve ser preservado para manter as versões das dependências.

Os resultados podem variar entre execuções devido à inicialização dos pesos, à aleatoriedade do treinamento, às transformações de augmentation e às diferenças de hardware.

A avaliação quantitativa apresentada pelo programa utiliza patches do conjunto de teste da seção original. Já a aplicação em outra seção permite uma análise visual do comportamento do modelo em dados não utilizados durante o treinamento.

---

## 9. Referência acadêmica

Este projeto está relacionado a uma pesquisa de mestrado sobre classificação de estruturas geológicas em imagens sísmicas utilizando aprendizado profundo.

A fundamentação teórica, os detalhes da metodologia e a discussão dos experimentos estão apresentados na dissertação correspondente.

---

## 10. Licença e utilização

Este repositório é disponibilizado para fins de pesquisa e reprodução acadêmica.

As condições de reutilização do código devem ser verificadas na licença adotada pelo projeto. Os dados sísmicos e modelos treinados podem estar sujeitos a condições de distribuição específicas.


## Autor

**Thiago Theiry de Oliveira**

Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPgEEC)  
Universidade Federal do Rio Grande do Norte (UFRN)

