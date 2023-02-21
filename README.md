# qos
Quantum Operating System

# File organisation

## backends/
Source code for running tasks on each different backend (ibmq-simulators, ibmq-physical, classical-resources, etc...).
Some code might be specific to QVM and or reference QVM code that was taken out and is now inside the *might_be_useful* folder.

## benchmarks/
Bechmarking scripts to measure different
Some code might be specific to QVM and or reference QVM code that was taken out and is now inside the *might_be_useful* folder.

## tests/
Test scripts for debugging proposes.
Some code might be specific to QVM and or reference QVM code that was taken out and is now inside the *might_be_useful* folder.

## application_layer/
Scrips and code left from the qvm project. Might be useful in the future, but this code is not included on the QOS code base.

## qos
Contains the code for the QOS components.

## Dockerfile
Used to start a docker container which should meet every dependency. Run everything inside this container. You can even run the
container and work directly on the container since the file are set be shared between the host machine and the container.

## README.md
This file


# Benchmark Arguments
Arg* - Means that the argument is not optional

## FermionicSwapQAOABenchmark
- number of qbits (`nqbits`)*

## VanillaQAOABenchmark
- number of qbits (`nqbits`)*

## GHZBenchmark
- number of qbits (`nqbits`)*

## MerminBellBenchmark
- number of qbits (`nqbits`)*

## VQEBenchmark
- number of qbits (`nqbits`)*
- number of layers (`nlayers`)
    Apparent change: The circuit becomes longer (same number of qbits, still outputs two circuits but each circuit is almost `nlayers` times as large)

## HamiltonianSimulationBenchmark
- number of qbits (`nqbits`)*
- time step (`time_step`)
- total time (`total_time`)
    Apparent change: The time step needs to be changed in conjuntion with the total time. The circuit is copied and concatenated, with measurement only in the end, `total_time//time_step` times

## BitCodeBenchmark
- number of qbits (`nqbits`)*
- rounds (`rounds`)*
- initial state (`initial_state`)

## PhaseCodeBenchmark
- number of qbits (`nqbits`)*
- rounds (`rounds`)*
- initial state (`initial_state`)

