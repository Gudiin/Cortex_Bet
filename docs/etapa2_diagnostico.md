# Etapa 2 — Diagnóstico Técnico (sem refatoração grande)

Data: 2026-05-12

## 1) Bugs prováveis

### Problema encontrado
Pipeline de treino pode bloquear em execução não interativa ao cair em `input()` para escolher modo.

### Local provável
`src/training/trainer.py`, função `train_model`, trecho de seleção de modo.

### Risco
Execuções automáticas (API/cron/scanner) podem travar indefinidamente.

### Solução sugerida
Definir modo explícito por argumento/env e fallback seguro sem prompt em contexto headless.

### Alteração aplicada ou proposta
**Proposta** (não aplicada nesta etapa): adicionar detecção `stdin.isatty()` e default configurável.

### Nível de risco da alteração
Baixo.

### Precisa de validação humana?
Não.

---

### Problema encontrado
Possível inconsistência temporal em parte do fluxo neural por uso de `train_test_split(..., shuffle=False)` após transformação sem validação explícita de ordenação global.

### Local provável
`src/ml/train_neural.py`.

### Risco
Data leakage sutil e métricas infladas se a série não estiver estritamente ordenada por timestamp em todos os pontos.

### Solução sugerida
Garantir ordenação por timestamp única antes de qualquer split e logar intervalo temporal de treino/teste.

### Alteração aplicada ou proposta
**Proposta**.

### Nível de risco da alteração
Médio.

### Precisa de validação humana?
Sim (revisão de protocolo científico).

---

### Problema encontrado
Loop do scanner tem blocos com `pass` e lógica incompleta para cálculo do próximo jogo.

### Local provável
`scripts/quick_scan.py` (smart interval).

### Risco
Intervalo subótimo (mais chamadas que o necessário ou menor responsividade).

### Solução sugerida
Concluir cálculo com timestamp real dos jogos e eliminar placeholders.

### Alteração aplicada ou proposta
**Proposta**.

### Nível de risco da alteração
Baixo.

### Precisa de validação humana?
Não.

## 2) Repetições e duplicações

- Há múltiplas estratégias de treino (CLI, script de treino, treino neural dedicado) com decisões de parâmetros e logging parcialmente duplicadas: `src/training/trainer.py`, `scripts/train_model.py`, `src/ml/train_neural.py`.
- Há lógica de distribuição probabilística replicada entre módulos de inferência neural e manager (`Poisson/NegBin`).

## 3) Problemas de arquitetura

- Orquestração mistura responsabilidades de API, processo e automação no mesmo backend (`src/api/server.py`).
- Scanner contém responsabilidade de coleta, resolução de apostas, cleanup e agora gatilho de treino no mesmo loop (`scripts/quick_scan.py`).
- Falta um “service layer” dedicado para jobs assíncronos (scanner/training scheduler).

## 4) Problemas de performance

- Scanner busca detalhes de cada partida de forma sequencial.
- Recarregamento de histórico completo em fluxos frequentes pode ser custoso (`ManagerAI` e treino).
- Possível recomputação de features sem cache temporal por janela.

## 5) Riscos no modelo (ML/neural)

- Necessidade de validação temporal formal (walk-forward/backtest em blocos).
- Ausência de baseline explícito no relatório operacional (ex.: média móvel / Poisson simples comparativa).
- Cap de confiança fixo e blend heurístico exigem calibração contínua documentada.

## 6) Riscos nos cálculos de escanteios

- Mistura de sinais HT/ST/FT é boa, mas exige checagem de completude (NULLs tratados como 0 podem enviesar média).
- Separação casa/fora precisa ser auditada por amostragem com partidas reais por time.

## 7) Foco em validade científica (pedido do usuário)

Pontos **alinhados** a práticas comuns:
- Uso de Poisson deviance para alvo de contagem no tuning neural.
- Separação temporal sem shuffle no treino.
- Combinação Poisson/NegBinom para overdispersion.

Pontos que **ainda exigem validação humana**:
- Protocolo formal de avaliação (janela temporal fixa + rolling origin).
- Relatório de calibração probabilística (ECE/Brier por faixa temporal).
- Teste estatístico de significância contra baseline (ex.: Diebold-Mariano em erro temporal).

## 8) Lista de correções seguras para próxima etapa (Etapa 3)

1. Remover blocos `pass` remanescentes do scanner e fechar lógica de `next_start`.
2. Tornar treino não interativo por padrão quando chamado via endpoint/job.
3. Adicionar logs estruturados para treino disparado por API (PID, start/end, status).
4. Criar testes unitários para validação de recência casa/fora (últimos 5).
5. Adicionar teste de regressão para prevenção de leakage temporal no treino neural.

## 9) Itens que exigem validação humana

- Definição final da janela de re-treino (15 dias) por competição/volume de novos dados.
- Critério oficial de promoção challenger→champion (métrica primária e mínimo de amostra).
- Escolha do benchmark oficial para comprovação científica em produção.


## 10) Verificação solicitada: `train_joint_model()` / JointCornersModel

Status atual observado no código: **não encontrado/ativo**.

- Não existe definição `def train_joint_model()` nos módulos operacionais inspecionados.
- Não há referência a `JointCornersModel` no pipeline de treino atual.
- O menu atual de treino expõe modos padrão/Optuna/Transfer/Neural Challenger, mas não o modo joint multimercado (4 targets h1H,a1H,h2H,a2H).

Implicação:
- Hoje o runtime **não está usando** esse treino multimercado científico descrito.
- Se este é o modo desejado de produção, precisa entrar como modo explícito no trainer + script operacional + testes de contrato.
