# qos
Quantum Operating System


# TODO

- [ ] Database should store a list with the ids of the current jobs
- [ ] Implement window on the database (initWindow, getCurrentWindow, moveWindow, ...)
- [ ] There should be a way concurrently running the scheduler and executing the circuits on the cloud


# Desing decisions

Should the matching engine receive the whole job, schedule the subjobs and then send the whole jobs to the multiprogramming engine
OR
The matching engine receives each subjob from the transformer and then submits each subjob to the multiprogamming engine?
## Arguments
- If the matching engine receives the whole job every time we want to match a single job we would need to send a job with a single subjob, which is not friendly
- If the matching engine only receives subjobs it will send subjobs to the multiprogrammer which will only consider merging a single subjob with one of the jobs on the window and not the other subjobs

---

The storing on the database is being done with the hash data type from redis. This works, however seems kind sketchy the way that we are converting from database data to object.
A Redis hash is basicly a dictionary of strings (a set of string keys with string values).
Now, on the database adding values to the hash set is relatively straightforward, then converting the hashset to an object we have to convert each key string from the Redis dict to a object attribute
OR just copy the whole redis dict as a python dictionary of job arguments, which is not that user friendly.

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
The same applies to the jobs, job hashes and "jobList"

The database will store three structures: Jobs, Circuits and QPUs

### Jobs


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