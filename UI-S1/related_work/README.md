# Related Work: Long-Horizon Reasoning in GUI Agents

Papers whose primary lever is **reasoning over a long horizon** in GUI and computer-use agents.

## Core Papers

| Paper | Venue | arXiv | GitHub | Key Mechanism |
|-------|-------|-------|--------|---------------|
| HAR | AAAI 2026 | [2511.09127](https://arxiv.org/abs/2511.09127) | [BigTaige/HAR-GUI](https://github.com/BigTaige/HAR-GUI) | In-trace history-aware reasoning + hybrid RL reward |
| AndroTMem | Preprint | [2603.18429](https://arxiv.org/abs/2603.18429) | [CVC2233/AndroTMem](https://github.com/CVC2233/AndroTMem) | Anchored state memory + causal-link retrieval |
| AgentProg | MobiSys 2026 | [2512.10371](https://arxiv.org/abs/2512.10371) | [MobileLLM/AgentProg](https://github.com/MobileLLM/AgentProg) | Semantic Task Program + belief state (Belief-MDP) |
| LongHorizonUI | ICLR 2026 (under review) | [OpenReview](https://openreview.net/forum?id=BK7Mk5d4WE) | [kane2kang/LongHorizonUI](https://github.com/kane2kang/LongHorizonUI) | Perceive + reflect + compensate (rollback) |
| HiMAC | Preprint | [2603.00977](https://arxiv.org/abs/2603.00977) | Not released | Macro-micro hierarchy + critic-free bi-level RL |
| CoMEM (Auto-Scaling) | Preprint | [2510.09038](https://arxiv.org/abs/2510.09038) | [WenyiWU0111/CoMEM-Agent](https://github.com/WenyiWU0111/CoMEM-Agent) | Fixed-length continuous memory embeddings |
| Mirage-1 | Preprint | [2506.10387](https://arxiv.org/abs/2506.10387) | [JiuTian-VL/Mirage-1](https://github.com/JiuTian-VL/Mirage-1) | Hierarchical multimodal skills + SA-MCTS |

## Adjacent Papers

| Paper | arXiv | GitHub | Angle |
|-------|-------|--------|-------|
| AgentFold | [2510.24699](https://arxiv.org/abs/2510.24699) | Not released | Proactive context folding for long-horizon web |
| UI-TARS-2 | [2509.02544](https://arxiv.org/abs/2509.02544) | [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS) | MoE-Transformer native agent, multi-turn RL |
| Mobile-Agent-v3 / GUI-Owl | [2508.15144](https://arxiv.org/abs/2508.15144) | [X-PLUG/MobileAgent](https://github.com/X-PLUG/MobileAgent) | End-to-end foundation agent + hierarchical framework |
| NaturalGAIA | [2508.01330](https://arxiv.org/abs/2508.01330) | [KeLes-Coding/NatureGAIA](https://github.com/KeLes-Coding/NatureGAIA) | Verifiable benchmark, macro planning vs micro execution |
| DART | [2602.00994](https://arxiv.org/abs/2602.00994) | [sheriyuo/DART](https://github.com/sheriyuo/DART) | Disjoint-LoRA routing for reasoning vs tool-use |

## Survey

| Paper | arXiv | GitHub |
|-------|-------|--------|
| GUI Agents with RL: Toward Digital Inhabitants | [2604.27955](https://arxiv.org/abs/2604.27955) | [Steve2457/Awesome-RL-GUI-Agents](https://github.com/Steve2457/Awesome-RL-GUI-Agents) |

## Directory Structure

```
related_work/
├── README.md              # This file
├── agentprog/             # Our V22c adaptation of AgentProg's STP + Belief State
│   ├── prompts.py
│   ├── helpers.py
│   ├── eval_agentprog.py
│   └── scripts/
│       └── eval_agentprog.slurm
├── har/                   # TODO
├── androtmem/             # TODO
├── longhorizonui/         # TODO
├── himac/                 # TODO
├── comem/                 # TODO
└── mirage1/               # TODO
```
