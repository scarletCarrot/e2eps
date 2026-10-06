# Assumptions

The brief leaves most system parameters open. These assumptions let the work move forward. Each one is configurable in the scenario file, so changing it later is a config edit, not a code change.

## 1. Constellation and orbits

| # | Assumption | Justification |
|---|---|---|
| A1 | LEO broadband constellation, Walker-Delta shell, reference ~550 km altitude, 53° inclination | Most common commercial LEO broadband geometry; low latency and wide mid-latitude coverage |
| A2 | Propagation: two-body + J2 secular perturbation (RAAN and argument-of-latitude drift) | J2 dominates LEO error over hours to days. Full SGP4/HPOP is overkill for capacity studies; SGP4 can be plugged in later behind the same interface |
| A3 | Circular orbits | Operational LEO broadband shells are near-circular; simplifies the generator |
| A4 | Earth rotation via GMST only (no polar motion, nutation) | Error is metres, irrelevant at the link-budget level |

## 2. Ground and user segment

| # | Assumption | Justification |
|---|---|---|
| A5 | Ground positions on WGS84 ellipsoid | Standard; cheap to compute once |
| A6 | Fixed user terminals (no mobility in v1) | Covers residential/enterprise use. Maritime/aero mobility is an extension of the same arrays |
| A7 | Minimum elevation: 25° user terminals, 10° gateways | Typical operator masks; low elevation means long slant range, high atmospheric loss and terrain blockage |
| A8 | Bent-pipe payload: a user is served only if its satellite also sees a gateway | Default for many current systems. Inter-satellite links (ISL) are out of scope for v1 |

## 3. RF link

| # | Assumption | Justification |
|---|---|---|
| A9 | Focus on the **user downlink** (satellite to terminal), Ku-band ~11.7 GHz | Usually the capacity bottleneck of a broadband system. Uplink and feeder links reuse the same engine with different parameters |
| A10 | Losses: free-space path loss + gaseous (fixed) + rain (simplified ITU-R P.838 / P.618 approach) + pointing/scan loss | Captures the effects that move the needle on throughput; scintillation and cloud ignored in v1 |
| A11 | Satellite phased array with scan loss growing with off-boresight angle | Phased arrays lose gain when steering away from nadir; this is the main reason low-elevation users get less throughput |
| A12 | C/N only - **no interference** (C/(N+I)) in v1 | Interference needs frequency plan and beam layout we don't have. Flagged as a top question |
| A13 | Adaptive coding and modulation using a DVB-S2X MODCOD table, plus implementation margin | Industry standard for broadband satellite; maps Es/N0 to spectral efficiency |

## 4. Capacity, availability and latency

| # | Assumption | Justification |
|---|---|---|
| A14 | Each terminal connects to the highest-elevation visible satellite | Simple, common baseline policy; real scheduler can replace it |
| A15 | Satellite user-link bandwidth shared equally among its connected terminals | Fair-share approximation of a real scheduler; good enough for capacity trends |
| A16 | Link "available" when Es/N0 >= lowest MODCOD threshold + margin | Standard availability definition for ACM links |
| A17 | Latency = propagation delay (terminal - satellite - gateway) + fixed processing/terrestrial budget | E2EPS is physics-level by design; queuing delay belongs to the packet-level emulator in the Digital Twin |

## 5. Simulation and compute

| # | Assumption | Justification |
|---|---|---|
| A18 | Time-stepped simulation, default 30 s step | Satellite moves ~225 km per step; fine for KPI statistics. Handover-level studies can use smaller steps |
| A19 | Data held as NumPy arrays shaped (time, satellite, ground point), processed in time chunks | Addresses the observed compute limits: vectorised maths instead of Python loops, bounded memory |
| A20 | Single-machine execution in v1; chunks are independent so they can be distributed (Dask/Ray) later | Right-sized for the team; scale-out path is designed in, not built |
