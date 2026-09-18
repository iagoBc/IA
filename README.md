# Inteligência Artificial — projetos CS 188

Este repositório reúne implementações e experimentos da disciplina de
Inteligência Artificial baseados nos **Pac-Man Projects** do curso **CS 188 —
Introduction to Artificial Intelligence**, da University of California,
Berkeley.

O material oficial dos projetos está disponível em:
[CS 188 — The Pac-Man Projects](https://inst.eecs.berkeley.edu/~cs188/sp26/projects/).

## Projetos

### Project 1 — Search

Implementação de algoritmos clássicos de busca para resolver problemas no
mundo do Pacman:

- busca em profundidade (DFS);
- busca em largura (BFS);
- busca de custo uniforme (UCS);
- busca A*;
- agentes de busca e problemas de planejamento no arquivo `searchAgents.py`.

Os arquivos deste projeto estão em [`search/`](./search/).

### Project 2 — Multiagent Search

Implementação de agentes que tomam decisões em um ambiente adversarial ou
estocástico:

- agente reflexo;
- busca minimax;
- poda alfa-beta;
- busca expectimax;
- funções de avaliação.

Os arquivos deste projeto estão em [`multiagent/`](./multiagent/), com as
principais implementações em
[`multiAgents.py`](./multiagent/multiAgents.py).

## Como executar

Os projetos usam Python 3 e não exigem dependências externas para a execução
básica. A partir da raiz do repositório, execute, por exemplo:

```bash
python3 search/pacman.py -l tinyMaze -p SearchAgent -a fn=bfs
python3 multiagent/pacman.py -p ReflexAgent -l testClassic
```

Para consultar as opções disponíveis:

```bash
python3 search/pacman.py -h
python3 multiagent/pacman.py -h
```

Cada projeto também contém seu autograder. Para executá-lo, consulte os
arquivos `autograder.py` correspondentes:

```bash
python3 search/autograder.py
python3 multiagent/autograder.py
```

## Créditos e atribuição

Os Pac-Man Projects foram desenvolvidos na UC Berkeley por John DeNero, Dan
Klein, Pieter Abbeel e colaboradores. O código-base e os autograders são
distribuídos para fins educacionais conforme os avisos presentes nos arquivos
do projeto. Consulte também o site original em
[ai.berkeley.edu](http://ai.berkeley.edu).

Este repositório é destinado a estudo acadêmico. Ao reutilizar o material,
mantenha os avisos de licença e a atribuição à UC Berkeley, e não distribua
soluções de trabalhos avaliativos quando isso contrariar as regras da
disciplina.
