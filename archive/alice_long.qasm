OPENQASM 2.0;
include "qelib1.inc";

qreg q[9];
creg c[8];

rz(3.14/6) q[0];
rz(3.14/6) q[1];
rz(3.14/6) q[2];
rz(3.14/6) q[3];
rz(3.14/6) q[4];
rz(3.14/6) q[5];
rz(3.14/6) q[6];
rz(3.14/6) q[7];
rz(3.14/6) q[8];

ry(3.14/6) q[0];
ry(3.14/6) q[1];
ry(3.14/6) q[2];
ry(3.14/6) q[3];
ry(3.14/6) q[4];
ry(3.14/6) q[5];
ry(3.14/6) q[6];
ry(3.14/6) q[7];
ry(3.14/6) q[8];

rz(3.14/6) q[0];
rz(3.14/6) q[1];
rz(3.14/6) q[2];
rz(3.14/6) q[3];
rz(3.14/6) q[4];
rz(3.14/6) q[5];
rz(3.14/6) q[6];
rz(3.14/6) q[7];
rz(3.14/6) q[8];

cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[6],q[7];

rz(3.14/6) q[2];
cx q[3],q[4];
cx q[7],q[5];
rz(3.14/6) q[6];
cx q[4],q[1];
ry(3.14/6) q[2];
rz(3.14/6) q[3];
rz(3.14/6) q[5];
ry(3.14/6) q[6];
rz(3.14/6) q[7];
rz(3.14/6) q[1];
rz(3.14/6) q[2];
ry(3.14/6) q[3];
rz(3.14/6) q[4];
ry(3.14/6) q[5];
rz(3.14/6) q[6];
ry(3.14/6) q[1];
rz(3.14/6) q[3];
ry(3.14/6) q[4];
rz(3.14/6) q[5];
rz(3.14/6) q[7];

cx q[5],q[7];
cx q[1],q[3];
cx q[2],q[4];
cx q[6],q[5];
cx q[3],q[1];
cx q[4],q[2];
cx q[7],q[6];
cx q[0],q[1];
cx q[3],q[6];
cx q[4],q[5];
cx q[7],q[8];

measure q[8] -> c[1];
measure q[5] -> c[3];
measure q[6] -> c[5];
measure q[1] -> c[7];
h q[0];
h q[3];
h q[4];
h q[7];
measure q[7] -> c[0];
measure q[4] -> c[2];
measure q[3] -> c[4];
measure q[0] -> c[6];