OPENQASM 2.0;
include "qelib1.inc";
qreg q1[6];
creg c1[6];
h q1[0];
h q1[1];
h q1[2];
h q1[3];
h q1[4];
x q1[5];
h q1[5];
barrier q1[0],q1[1],q1[2],q1[3],q1[4],q1[5];
cx q1[0],q1[5];
cx q1[2],q1[5];
cx q1[3],q1[5];
cx q1[4],q1[5];
barrier q1[0],q1[1],q1[2],q1[3],q1[4],q1[5];
h q1[0];
h q1[1];
h q1[2];
h q1[3];
h q1[4];
measure q1[0] -> c1[0];
measure q1[1] -> c1[1];
measure q1[2] -> c1[2];
measure q1[3] -> c1[3];
measure q1[4] -> c1[4];
