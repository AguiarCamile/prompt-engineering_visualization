# Repositório — Geração de Visualizações de Dados com LLMs

Este repositório reúne os principais artefatos utilizados para a
reprodução do experimento descrito na dissertação:

**Geração de Visualizações de Dados com Grandes Modelos de Linguagem:
um Estudo Experimental dos Efeitos do Uso de Exemplos e da Estrutura
Formal dos Prompts**

O objetivo é disponibilizar os materiais necessários para reproduzir
o procedimento experimental e os procedimentos automatizados utilizados
na avaliação das visualizações geradas.

## Estrutura do repositório

### `Imagem_referencia/`

Contém as visualizações de referência utilizadas no experimento,
correspondentes às seis combinações de técnica de visualização e
estrutura de identificação:

- BI — Barras com Identificação
- BC — Barras com Código
- LI — Linhas com Identificação
- LC — Linhas com Código
- SI — Scatterplot com Identificação
- SC — Scatterplot com Código

### `script_imagem_ref/`

Contém os scripts em Python utilizados para gerar as visualizações
de referência.

### `config_ambiente/`

Contém as configurações necessárias para a realização do experimento,
incluindo os prompts de inicialização, o arquivo visual, o dataset 
utilizado e exemplos utilizados nas diferentes condições experimentais.

### `avaliacao_visual/`

Contém os códigos e configurações utilizados para a avaliação
automatizada das visualizações geradas, incluindo:

- similaridade visual;
- similaridade textual e visual-semântica;
- conformidade visual;
- classificação das técnicas de visualização;
- requisitos e referências utilizados na avaliação.

### `Mapeamento_sistematico_Artigos/`

Contém um arquivo criado a partir do ParsiFal que apresenta os artigos
encontrados no mapeamento, se foram rejeitados ou aceitos em relação aos
critérios definidos.

## Procedimento experimental

O experimento combina os fatores relacionados ao uso de exemplos,
estrutura formal do prompt e cenários visuais (técnica de visualização 
e tarefa analítica), resultando em 24 condições experimentais por participante.

Em cada condição, o participante recebe a tarefa de reproduzir a imagem de 
referência, a determinada visualização de referência, e formula uma solicitação
ao modelo de linguagem. A visualização gerada pode ser avaliada e, quando 
necessário, o participante realiza refinamentos.

## Avaliação das visualizações

As visualizações geradas podem ser submetidas aos procedimentos
automatizados disponibilizados em `avaliacao_visual/`.

Os procedimentos incluem a comparação com as imagens de referência,
a análise semântica das visualizações e a verificação da conformidade
com os requisitos definidos.

## Referência

Este repositório está associado à dissertação de mestrado de Camile
Maria Cunha Aguiar, Programa de Pós-Graduação em Ciência da Computação (PPGCC),
Universidade Federal do Pará, 2026.
