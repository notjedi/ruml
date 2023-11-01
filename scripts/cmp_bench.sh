#!/bin/sh

git checkout 90067feec6be472b654ba4d8a948dfa543b2d7d6 src/backend/cpu.rs
RUSTFLAGS='-C target-feature=+avx2 -C target-cpu=native' cargo bench

git checkout 4954c866df709c41e284ccf3ee0a3213ab71293a src/backend/cpu.rs
RUSTFLAGS='-C target-feature=+avx2 -C target-cpu=native' cargo bench

git reset
git restore src/backend/cpu.rs
