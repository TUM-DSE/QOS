OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c5[16];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
measure q[0] -> c5[0];
measure q[1] -> c5[1];
measure q[2] -> c5[2];
measure q[3] -> c5[3];
measure q[4] -> c5[4];
measure q[5] -> c5[5];
measure q[6] -> c5[6];
measure q[7] -> c5[7];
measure q[8] -> c5[8];
measure q[9] -> c5[9];
measure q[10] -> c5[10];
measure q[11] -> c5[11];
measure q[12] -> c5[12];
measure q[13] -> c5[13];
measure q[14] -> c5[14];
measure q[15] -> c5[15];
