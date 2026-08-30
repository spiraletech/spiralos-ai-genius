# Spiral L24 Performance Proof

This branch exists to replace architecture claims with reproducible measurements.

Proof gates:
1. End-to-end transformer-style attention executes as a GPU-resident chain on Windows D3D11.
2. GPU and CPU reference outputs satisfy numerical parity thresholds.
3. Model weights remain resident across repeated forwards; transfer counters prove no hidden CPU bounce between GPU ops.
4. A benchmark executable reports CPU time, GPU cold time, GPU-resident time, adapter identity, hardware-vs-WARP status, numerical error, dispatch count, and transfer bytes.
5. CI publishes benchmark output as an artifact. Hosted WARP results are labeled compatibility results and are never claimed as physical-GPU speedups.
6. No performance win is claimed unless measured on hardware_accelerated=true.
