# qos
Quantum Operating System

# * Log

- The refactoring seems to be fine up until the Optimiser call of the Matcher - Continue here 




# TODO - QOS System V2 refactoring

- [ ] Rename jobs to Qernels
- [ ] Redo the Qernel properties
- [ ] Create an AST class
- [ ] Rename and reorganize folders into system blocks?
- [ ] Create the Analyser component
- [ ] Extend the DAG code to work with other libraries or to be library independent (now it works with qiskit)

# Other TODOS

- [ ] Database should store a list with the ids of the current qernels
- [ ] Implement window on the database (initWindow, getCurrentWindow, moveWindow, ...)
- [ ] There should be a way concurrently running the scheduler and executing the circuits on the cloud


# Design decisions

Should the matching engine receive the whole qernel, schedule the subqernels and then send the whole qernels to the multiprogramming engine
OR
The matching engine receives each subqernel from the transformer and then submits each subqernel to the multiprogamming engine?
## Arguments
- If the matching engine receives the whole qernel every time we want to match a single qernel we would need to send a qernel with a single subqernel, which is not friendly
- If the matching engine only receives subqernels it will send subqernels to the multiprogrammer which will only consider merging a single subqernel with one of the qernels on the window and not the other subqernels

---

The storing on the database is being done with the hash data type from redis. This works, however seems kind sketchy the way that we are converting from database data to object.
A Redis hash is basicly a dictionary of strings (a set of string keys with string values).
Now, on the database adding values to the hash set is relatively straightforward, then converting the hashset to an object we have to convert each key string from the Redis dict to a object attribute
OR just copy the whole redis dict as a python dictionary of qernel arguments, which is not that user friendly.

---



# Logging

Logging something from information messages to critical errors uses the `logging` python library.
The following logging levels are used.

![logging levels](https://www.loggly.com/wp-content/uploads/2022/09/logging-levels.png)

# Development Log

## Database

The database uses a Redis database, before running QOS the redis server needs to be started with `redis-server ./redis.conf`

Since the database data is stored in a file in the background and at exit to clean the database do (with the redis server running):

```
redis-cli
flushall
```

The database stores all the available QPUs in hash datastrutures, the hashes identification keys are the QPU's ids. To easly find unsued ids, they are also store is a set "qpuList".
The same applies to the qernels, qernel hashes and "qernelList"

The database will store three structures: Qernels and QPUs

### Qernels


---





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