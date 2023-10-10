OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.88853126) q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
rz(0.88853126) q[2];
cx q[1],q[2];
rx(8.6314465) q[2];
h q[3];
cx q[0],q[3];
rz(0.88853126) q[3];
cx q[0],q[3];
rx(8.6314465) q[0];
rx(8.6314465) q[3];
h q[4];
cx q[1],q[4];
rz(0.88853126) q[4];
cx q[1],q[4];
h q[5];
cx q[1],q[5];
rz(-0.88853126) q[5];
cx q[1],q[5];
rx(8.6314465) q[5];
h q[6];
cx q[1],q[6];
rz(0.88853126) q[6];
cx q[1],q[6];
rx(8.6314465) q[1];
rx(8.6314465) q[6];
h q[7];
cx q[4],q[7];
rz(-0.88853126) q[7];
cx q[4],q[7];
rx(8.6314465) q[4];
rx(8.6314465) q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];

