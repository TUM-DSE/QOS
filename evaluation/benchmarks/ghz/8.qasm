OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c1[8];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
measure q[0] -> c1[0];
measure q[1] -> c1[1];
measure q[2] -> c1[2];
measure q[3] -> c1[3];
measure q[4] -> c1[4];
measure q[5] -> c1[5];
measure q[6] -> c1[6];
measure q[7] -> c1[7];
