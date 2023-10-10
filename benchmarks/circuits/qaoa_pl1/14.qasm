OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
h q[0];
h q[1];
cx q[0],q[1];
rz(-5.2042545) q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
rz(5.2042545) q[2];
cx q[1],q[2];
rx(7.10825) q[2];
h q[3];
cx q[1],q[3];
rz(-5.2042545) q[3];
cx q[1],q[3];
rx(7.10825) q[3];
h q[4];
cx q[1],q[4];
rz(-5.2042545) q[4];
cx q[1],q[4];
rx(7.10825) q[4];
h q[5];
cx q[1],q[5];
rz(5.2042545) q[5];
cx q[1],q[5];
rx(7.10825) q[5];
h q[6];
cx q[1],q[6];
rz(-5.2042545) q[6];
cx q[1],q[6];
h q[7];
cx q[1],q[7];
rz(-5.2042545) q[7];
cx q[1],q[7];
h q[8];
cx q[0],q[8];
rz(5.2042545) q[8];
cx q[0],q[8];
rx(7.10825) q[0];
rx(7.10825) q[8];
h q[9];
cx q[7],q[9];
rz(-5.2042545) q[9];
cx q[7],q[9];
rx(7.10825) q[9];
h q[10];
cx q[1],q[10];
rz(5.2042545) q[10];
cx q[1],q[10];
rx(7.10825) q[10];
h q[11];
cx q[7],q[11];
rz(5.2042545) q[11];
cx q[7],q[11];
rx(7.10825) q[11];
rx(7.10825) q[7];
h q[12];
cx q[1],q[12];
rz(5.2042545) q[12];
cx q[1],q[12];
rx(7.10825) q[1];
rx(7.10825) q[12];
h q[13];
cx q[6],q[13];
rz(5.2042545) q[13];
cx q[6],q[13];
rx(7.10825) q[13];
rx(7.10825) q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
