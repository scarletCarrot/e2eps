# Implemented Modules - Inputs, Outputs and Why They Matter

The brief asks to defend the inputs and outputs of the implemented modules. Both modules sit on the critical path from "where are the satellites" to "what does the user get", so every KPI E2EPS reports depends on them.

```
Orbit Propagator -> [A] Geometry & Visibility -> [B] RF Link Budget -> Capacity -> KPIs
                                \-> Latency ------------------------------------/
```

## Module A - Geometry & Visibility Engine

`e2eps/geometry/engine.py` - `GeometryEngine.compute(sat_ecef) -> GeometryResult`

### Inputs

| Input | Shape / unit | Source | Why it is needed |
|---|---|---|---|
| Satellite positions (Earth-fixed) | `(T, S, 3)` m | Orbit Propagator | Where every satellite is at every time step |
| Ground positions (WGS84 -> ECEF) | `(G, 3)` m | Scenario (terminals / gateways) | Where users and gateways are |
| Local vertical ("up") per ground point | `(G, 3)` | Derived once from lat/lon | Elevation is measured from the local horizon, not from Earth's centre |
| Minimum elevation mask | `(G,)` deg | Scenario (A7) | Below it, terrain, buildings and long atmospheric paths make links unusable |
| Max satellite scan angle (optional) | deg | Scenario (A11) | A phased array cannot steer beyond its limit |

### Outputs

| Output | Shape / unit | Used by | Why it is critical |
|---|---|---|---|
| Elevation angle | `(T, S, G)` deg | Visibility, RF (gas, rain), allocation | Decides if a link exists; low elevation = long atmospheric path = higher fade; selection policy uses it |
| Slant range | `(T, S, G)` m | RF (free-space loss), latency | Free-space loss is the largest term in the budget (~169-175 dB); range also sets propagation delay |
| Off-nadir (scan) angle | `(T, S, G)` deg | RF (scan loss) | Phased-array gain drops as the beam steers away from nadir - main reason edge users get less |
| Visibility mask | `(T, S, G)` bool | RF, allocation, coverage KPI | Removes impossible links before any RF work (typically > 95% of pairs) |
| Visible count | `(T, G)` | Coverage KPI | Coverage depth = resilience and handover options |

### Why it matters for the brief

- It is the **computational hotspot** (~80% of runtime) and the direct answer to "computing limitations". The BLAS reformulation is ~3x faster and uses ~3x less memory than plain NumPy broadcasting, ~90-190x faster than a Python loop (`benchmarks/results.md`).
- Same engine serves **terminals and gateways** (bent-pipe feeder check and latency).
- Validated against analytic geometry (overhead, horizon, spherical law of sines) and two reference implementations.

## Module B - RF Link Budget Engine

`e2eps/rf/link_budget.py` - `LinkBudgetEngine.compute(geometry) -> LinkResult`

### Inputs

| Input | Source | Why it is needed |
|---|---|---|
| Elevation, slant range, off-nadir, visibility | Module A | Geometry drives every loss term |
| Frequency | Scenario | Free-space and rain loss scale with frequency (Ku vs Ka is a design decision) |
| Satellite EIRP at boresight | Scenario (payload design) | Transmitted power - the main capacity lever |
| Scan-loss exponent, max scan | Scenario (antenna design) | Phased-array performance vs steering angle |
| Terminal G/T | Scenario (terminal design) | Receiver sensitivity - differs per terminal type |
| Channel bandwidth, roll-off | Scenario (frequency plan) | Converts Es/N0 and spectral efficiency into bit/s |
| Gaseous zenith loss, rain rate, rain height | Scenario (climate) | Atmospheric fades - drive availability |
| Misc losses, implementation margin | Scenario | Realistic margins (pointing, polarisation, modem) |
| MODCOD table | DVB-S2 (A13) | Maps link quality to achievable efficiency |

### Outputs

| Output | Shape | Used by | Why it is critical |
|---|---|---|---|
| Es/N0 | `(T, S, G)` dB | MODCOD selection, reports | Single number that captures the physics of the link |
| MODCOD index | `(T, S, G)` | KPIs, plots | Shows how ACM trades robustness for efficiency |
| Spectral efficiency | `(T, S, G)` bit/symbol | Link rate | Converts physics into capacity |
| Link rate | `(T, S, G)` bit/s | Capacity allocator | Upper bound per link before sharing - the basis of throughput and capacity KPIs |
| Available flag | `(T, S, G)` bool | Allocation, availability KPI | Visible **and** closes at least the most robust MODCOD |
| Per-link breakdown | single link | RF engineers | Classic budget table to verify against their own tools |

### Why it matters for the brief

- It is where **capacity, throughput and availability come from**. Without it, geometry only tells you "a satellite is visible", not "what the user gets".
- Rain what-if shows the value: at 50 mm/h, availability stays at 100% but P5 throughput drops ~70% because ACM falls back to robust MODCODs (`docs/results/`).
- Evaluates only visible links, so cost scales with real links, not all pairs.

## What the results say (reference scenario)

1. **Throughput is limited by sharing, not by the link.** ~87% of served samples already use the top MODCOD; capacity tracks the number of active satellites (~18 over Europe) and ~14 terminals share each one. The model assumes one channel per satellite - **number of beams per satellite** is the most important missing input (questions Q8, Q19).
2. **Edge-of-coverage users** lose ~10 dB vs nadir (path + scan + atmosphere) - visible as the latitude banding in the coverage map.
3. **Latency is ~20 ms one-way** with 15 ms of it fixed processing/backhaul - the gateway network and backhaul assumptions matter more than orbit geometry (Q13).
4. **~50 handovers/hour** per terminal with the highest-elevation policy - worth reviewing with Network Engineering (Q12).
