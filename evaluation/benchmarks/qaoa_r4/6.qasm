OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c40[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
rzz(0.1233983180273463) q[0],q[1];
rzz(0.1233983180273463) q[0],q[2];
rzz(0.1233983180273463) q[0],q[4];
rzz(0.1233983180273463) q[0],q[5];
rzz(0.1233983180273463) q[1],q[3];
rzz(0.1233983180273463) q[1],q[4];
rzz(0.1233983180273463) q[1],q[5];
rzz(0.1233983180273463) q[2],q[3];
rzz(0.1233983180273463) q[2],q[4];
rzz(0.1233983180273463) q[2],q[5];
rzz(0.1233983180273463) q[3],q[5];
rzz(0.1233983180273463) q[4],q[3];
rx(2.0666072699874287) q[0];
rx(2.0666072699874287) q[1];
rx(2.0666072699874287) q[2];
rx(2.0666072699874287) q[3];
rx(2.0666072699874287) q[4];
rx(2.0666072699874287) q[5];
measure q[0] -> c40[0];
measure q[1] -> c40[1];
measure q[2] -> c40[2];
measure q[3] -> c40[3];
measure q[4] -> c40[4];
measure q[5] -> c40[5];
