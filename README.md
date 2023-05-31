# qos
Quantum Operating System

# Logging

Logging something from information messages to critical errors uses the `logging` python library.
The following logging levels are used.

![logging levels](https://www.loggly.com/wp-content/uploads/2022/09/logging-levels.png)













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

# Log

## Draft

### Evalutation - Challenges

#### Introducion
This is an introduction the actual evaluation which are the challenges. I think it is at least drafted. It is already adapted to the 5 challenges.
- [x] Framworks
- [x] Backends
- [x] Benchmark
- [x] Performance Metrics

#### Challenges

##### Scalability
- [x] Results 
- [x] Implications 
- [x] Proposal 
- [x] Effectiveness 
The Scalability challenge is drafted, the two figures are there and are explained on the Results and Effectiveness sections.

##### Efficiency
- [x] Results 
- [x] Implications 
- [x] Proposal 
- [x] Effectiveness 
The effienciency challenges is drafted, the two figures are good and clearly depict the effectiveness of the solution.

##### Spatial Hereginity
- [x] Results 
- [x] Implications 
- [x] Proposal 
- [ ] Effectiveness
The Effectiveness is missing because the figure is missing, but the rest should be enough for the draft.
    
##### Temporal Variance 
- [x] Results 
- [x] Implications 
- [x] Proposal 
- [ ] Effectiveness 
For this challenge the figure on for the Effectiveness is missing however, we are coming to the conclusion that the best machine usually is the best machine
throuhout time variance, which means that if we were to run on the best machine one day usually that is the best machine everyday and so there is no actuall challenge. Or is there?

##### Multiprograming
- [x] Results 
- [x] Implications 
- [x] Proposal 
- [ ] Effectiveness
The Effectiveness is also missing here, this is saying the plot showing results of the solution is missing. 