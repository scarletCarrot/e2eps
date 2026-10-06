# Open Questions for Engineering Teams

Grouped by team. Each question notes why it matters to E2EPS. Answers would replace assumptions in `assumptions.md`.

## Systems Engineering

1. **What decisions should E2EPS support?** (constellation sizing, gateway placement, SLA verification, customer sales quotes?) - drives required accuracy, run time and outputs.
2. **Which KPIs and percentiles define success?** (e.g. P95 throughput, 99.9% availability, P99 latency) - defines the KPI aggregator contract.
3. **Constellation definition** - number of shells, altitude, inclination, planes, satellites per plane, phasing. Is the source a design spec or live TLE/ephemeris? (A1-A3)
4. **Target scale and run time** - how many satellites, user points and time steps per run, and how fast must a run complete? - sizes the compute backend.
5. **Which accuracy is "good enough"?** Is there a reference tool or measured data to validate against?

## Radio Frequency

6. **Frequency plan** - bands for user and feeder links, channel bandwidth, frequency reuse scheme, polarisation. (A9)
7. **Interference** - is C/(N+I) required in v1? Intra-system (beam-to-beam) and inter-system (GEO arc avoidance, NGSO coordination)? (A12)
8. **Antenna models** - satellite array gain pattern and scan loss, terminal antenna gain and G/T, beam size and number of beams per satellite. (A11)
9. **Propagation models** - which ITU-R recommendations and climate data are mandated? Target rain availability (e.g. 99.5%)? (A10)
10. **Waveform** - DVB-S2X or proprietary? MODCOD thresholds and implementation margins. (A13)

## Network Engineering

11. **Payload architecture** - bent-pipe, regenerative, or ISL mesh? Changes how paths and latency are computed. (A8)
12. **Handover and satellite selection policy** - highest elevation, longest visibility, load-aware? (A14)
13. **Latency budget** - processing delays, gateway-to-PoP backhaul, where latency is measured to. (A17)
14. **Boundary with packet-level tools** - where should E2EPS hand over to network emulation (queuing, TCP, congestion)?

## Ground Segment

15. **Gateway locations, antennas and capacity** - feeder link may become the bottleneck at high load.
16. **User demand model** - terminal distribution (population, subscriber data), demand per terminal, busy-hour profile. (A15)
17. **Terminal types** - fixed, maritime, aero, mobile? Mobility changes data shapes and update rates. (A6)

## Space Segment / Payload

18. **Payload constraints** - power, thermal and duty-cycle limits that cap per-satellite throughput.
19. **Beam steering limits** - max scan angle, beam hopping, number of simultaneous beams.

## Software / Platform

20. **Current prototype** - where exactly are the bottlenecks (profiling data)? Loops over satellites/users, or heavy library calls?
21. **Integration** - who consumes E2EPS outputs (dashboards, other Digital Twin components, notebooks)? Preferred interfaces (REST, gRPC, files, message bus)?
22. **Infrastructure** - cloud provider, available compute (CPU/GPU clusters), data storage standards.
23. **Governance** - versioning of scenarios and results, traceability required for regulatory or customer reporting?
