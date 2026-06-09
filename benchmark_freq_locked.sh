#!/bin/bash

GT=/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0
sudo sh -c "echo 1200 > $GT/min_freq; echo 1200 > $GT/max_freq"
cargo bench -p benchmarks --bench dequantize
