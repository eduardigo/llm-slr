# Atualização de RSLs com LLMs open-weights

Projeto de dissertação de mestrado (UTFPR / PPGI) sobre classificação automática
de artigos na atualização de Revisões Sistemáticas da Literatura (RSLs), usando
LLMs open-weights rodando localmente via [Ollama](https://ollama.com).

O pipeline reaproveita os dados e os folds temporais da baseline SVM/TF-IDF de
Watanabe et al. (2020) (`slr-sentence-embedding-master`) e compara, fold a fold,
o desempenho dos LLMs (zero-shot e few-shot) contra essa baseline.

Autor: Eduardo Felipe Ardigo Braga.

## Estrutura do repositório

| Pasta | Conteúdo |
|-------|----------|
| `llm_slr/` | Pipeline principal em Python (dados, prompts, cliente LLM, avaliação, experimentos) |
| `slr-sentence-embedding-master/` | Repositório legado (baseline SVM/TF-IDF + arquivos `.bib` das RSLs) |
| `notebooks/` | Análise exploratória dos datasets (`01`) e dos resultados (`02`) |

Dentro de `llm_slr/`:

- `config.py` — temas/`.bib`, seed, folds temporais, escala Likert, modelos Ollama
- `data/` — carga dos `.bib` (`loader.py`) e split temporal (`splits.py`)
- `prompts/` — construção do prompt e critérios de inclusão por tema (`criteria/*.yaml`)
- `llm/` — cliente Ollama (`client.py`) e cache de respostas (`cache.py`)
- `eval/` — métricas (precision, recall, F1, WSS@95) e baseline SVM/TF-IDF
- `experiments/` — pontos de entrada executáveis (ver abaixo)
- `tests/` — testes unitários e o `test_parity.py`, que compara folds e loader
  com o código legado

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) rodando localmente (`http://localhost:11434`) com
  os modelos `llama3.2:3b`, `gemma2:2b` e `qwen2.5:3b`

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b && ollama pull gemma2:2b && ollama pull qwen2.5:3b
```

## Como usar

```bash
# Testes (unitários + paridade com o legado sobre os .bib reais)
python -m pytest llm_slr/tests -q

# Baseline SVM/TF-IDF (todos os temas) -> llm_slr/results/baseline_*.csv
python -m llm_slr.experiments.run_baseline

# Experimento principal: temas x modelos x estratégias x folds
python -m llm_slr.experiments.run
python -m llm_slr.experiments.run --themes slr,games --models llama3.2:3b

# Análise dos resultados
jupyter lab notebooks/
```

Os experimentos são retomáveis: combinações já gravadas em `results/raw.csv` são
puladas, e o cache de respostas (`llm_slr/cache/`, fora do versionamento) evita
pagar inferência de novo em reexecuções parciais.

## Decisões de projeto

- Escala de relevância de 1 a 7, com threshold varrido na avaliação, em vez de
  decisão binária
- Temperatura 0 e saída em JSON, para reprodutibilidade
- Prompt com critérios de inclusão apenas
- Exemplos few-shot amostrados do bloco de treino do fold temporal
- Métricas alinhadas às do `main.py` legado (`missed`/`excluded`), mais WSS@95

## Licença

O código e os artefatos de autoria própria (`llm_slr/`, `notebooks/`) são
distribuídos sob a licença [MIT](LICENSE).

O diretório `slr-sentence-embedding-master/` reproduz o repositório de
Watanabe et al. (2020) e **não** é coberto por essa licença: código, dados e
padrão ouro pertencem aos autores originais. Ver [NOTICE](NOTICE) para o
escopo detalhado.

## Créditos

Baseline e datasets derivados de Watanabe et al. (2020), *slr-sentence-embedding*,
mantidos em `slr-sentence-embedding-master/`.
