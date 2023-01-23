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

