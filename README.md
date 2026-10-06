# E2EPS - End-to-End Performance Simulator

Physics-level simulator for satellite constellation performance, part of the wider Digital Twin tool suite.

E2EPS estimates **capacity, user throughput, availability and latency** by modelling the physical layer of the system - orbital dynamics, line-of-sight geometry, dynamic RF link budgets and coverage - instead of packet-level network transport.

## Scope of this submission

| Deliverable | Location |
|---|---|
| Software architecture diagram (E2EPS) | `docs/` |
| Workflow diagram + data exchanged | `docs/` |
| Solutions architecture (Digital Twin) | `docs/` |
| Assumptions and questions for Engineering | `docs/` |
| Implemented modules (Python) | `e2eps/geometry`, `e2eps/rf` |

## Repository layout

```
docs/        architecture, workflow, solutions diagrams, assumptions, questions
e2eps/
  config/    scenario schema and loader
  orbit/     orbit propagator (feeds the geometry engine)
  geometry/  Module A - Geometry & Visibility Engine
  rf/        Module B - RF Link Budget Engine
  kpi/       KPI aggregation
scenarios/   sample scenario files
tests/       unit tests
benchmarks/  performance benchmarks
```

## Status

Work in progress - see Git history for step-by-step development.
