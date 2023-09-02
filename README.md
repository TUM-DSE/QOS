# qos
Quantum Operating System

# Log

- The refactoring seems to be fine up until the Optimiser call of the Matcher
- Problem: The dag is only initialized from a qiskit quantum circuit and to store it on the database I need to serialize the dag and then deserialize it. For this I need to create a DAG class from a DiGraph from the networkx library
    - Initilializing it from DiGraph its relatively easy but then the to_circuit doesnt work, I need to get the number of qregs and cregs from the DiGraph for the to_circuit to work.
    - Yup, I need to understand the dag's code.
    - I just found out that there is the pickle library which serializes python objects into python binary. Now this leads to design question 1.
- TODO: Do the struture of the virtualizer


# Design questions

1. The virtualizer is just one component. We can change this in the future but I think it makes sense to make the Error-Mitigation, and the "Virtualizations" as passes. The passes would be configurable in terms of their order and this order would be defined by the Optimizer. We could for example have multiple Error-Mitigation passes, one in the beggining and one in the end of the "Virtualizer". Even better the Optimizer wouldnt apply any techniques, it would be just the solver that would find the best techniques/passes to apply to the circuit and order them in a list. The Virtualizer would be the one that applies the passes.
    -   For example, the solver could output the following:
        [Gate_virtualization of gate 2,
         Gate_virtualizatino of gate 5,
         Error Mitigation technique X
         Instantiation virtual gate 2,
         Instantiation virtual gate 5]

    - Then the results would come back up and they would be processed by the Virtualizer in the reverse order of the transformation list. If one of the passes didnt require any modification of the results it would just have an empty "Knitter" function and be "transparent".

    - If someone wants to add a new technique, they just need to add the necessary passes to the "Virtualizer". The optimizer would also need to change its solver techinque but at least the passes the actually modify the circuit would be confined into a single component of the system. The Distributed Compiler would be just a static component.
    - We could rename the Virtualizer as Transformer, or the Optimizer as Solver or Optimization Solver and the Virtualizer as Optimizer, these some ideas.

2. Should we store on the database the pikle of the dag object, which is just a bunch of binary or an adjency graph? From both we can recontruct the dag object, the adj graph is way smaller but also has a small overhead to reconstruct the dag object, but probably this overhead is negligable.

3. The storing on the database is being done with the hash data type from redis. This works, however seems kind of sketchy the way that we are converting from database data to object. A Redis hash is basicly a dictionary of strings (a set of string keys with string values).
    - Now, on the database adding values to the hash set is relatively straightforward, then converting the hashset to an object we have to convert each key string from the Redis dict to a object attribute
    OR
    - Instead of having class attributes for the qernel we just have one attribute which is "args" and store there everything about the quernel, this way the args map directly to a redis hash.

4. About the question, should we send the whole qernel object from component to component or just the id and the next component would fetch the qernel object from the DB?
    - For now, this is how we handle this:
        1. The API creates the qernel object and stores it on the DB but still passes the whole object to the optimizer.
        2. The optimizer does its job, updates the object with the subqernels as whole objects and the DB with the subqernels as ids to the actual qernels on the DB, and passes again the whole qernel.
        3. The Matcher works on the subqernels without needing to fetch them from the DB since the whole subqernels are stored on the original qernel. The Matchings for the subqernels are updated on the DB and on the subqernels on the qernel object.
        4. Now the qernel object is piped to the Multiprogramming engine


# TODO - QOS System V2 refactoring

- [ ] Rename jobs to Qernels
- [ ] Redo the Qernel properties
- [ ] Create an AST class
- [ ] Rename and reorganize folders into system blocks?
- [ ] Create the Analyser component
- [ ] Extend the DAG code to work with other libraries or to be library independent (now
    it works with qiskit)

# Other TODOS

- [ ] Database should store a list with the ids of the current qernels
- [ ] Implement window on the database (initWindow, getCurrentWindow, moveWindow, ...)
- [ ] There should be a way concurrently running the scheduler and executing the circuits on the cloud


# Usefull information

## Logging

Logging something from information messages to critical errors uses the `logging` python library.
The following logging levels are used.

![logging levels](https://www.loggly.com/wp-content/uploads/2022/09/logging-levels.png)

Maybe we should have one to every logging level because on the debug levels it gives out a bunch of debug information from other libraries that we dont care. We can remove messages from other libraries but we need to do this for every individual library that we want to ignore.

## Database

The database uses a Redis database, before running QOS the redis server needs to be started with `redis-server ./redis.conf`

Since the database data is stored in a file in the background and at exit to clean the database do (with the redis server running):
```
redis-cli
flushall
```