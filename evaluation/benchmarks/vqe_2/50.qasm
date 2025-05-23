OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - 1.0*param1) q0; }
qreg q[50];
creg c36[50];
r(0.872810454506742,pi/2) q[0];
r(0.251059047047574,pi/2) q[1];
r(0.217201730326476,pi/2) q[2];
r(0.393023645689462,pi/2) q[3];
r(0.765776306022462,pi/2) q[4];
r(0.460790741944088,pi/2) q[5];
r(0.470133386622951,pi/2) q[6];
r(0.708703266133775,pi/2) q[7];
r(0.478203914491513,pi/2) q[8];
r(0.906608712532312,pi/2) q[9];
r(0.872702171455923,pi/2) q[10];
r(0.077438502246179,pi/2) q[11];
r(0.650305460828766,pi/2) q[12];
r(0.90843485841711,pi/2) q[13];
r(0.951280044232748,pi/2) q[14];
r(0.866248308268027,pi/2) q[15];
r(0.127965905806273,pi/2) q[16];
r(0.563823913027404,pi/2) q[17];
r(0.191550406332381,pi/2) q[18];
r(0.543964335532991,pi/2) q[19];
r(0.0275509419979313,pi/2) q[20];
r(0.442879151303218,pi/2) q[21];
r(0.684754603425892,pi/2) q[22];
r(0.69481674517847,pi/2) q[23];
r(0.624728938302412,pi/2) q[24];
r(0.096760901585071,pi/2) q[25];
r(0.444816809349624,pi/2) q[26];
r(0.461328793824722,pi/2) q[27];
r(0.272983383929282,pi/2) q[28];
r(0.0532942080623583,pi/2) q[29];
r(0.872499639793517,pi/2) q[30];
r(0.0988072847820711,pi/2) q[31];
r(0.446924774795586,pi/2) q[32];
r(0.507981294425177,pi/2) q[33];
r(0.415462354181377,pi/2) q[34];
r(0.924264659350279,pi/2) q[35];
r(0.147077968917523,pi/2) q[36];
r(0.0455888811871942,pi/2) q[37];
r(0.671483793564765,pi/2) q[38];
r(0.575365625050177,pi/2) q[39];
r(0.56904507070922,pi/2) q[40];
r(0.946534566726024,pi/2) q[41];
r(0.710017510084434,pi/2) q[42];
r(0.869100974981507,pi/2) q[43];
r(0.485671889497658,pi/2) q[44];
r(0.707448106268686,pi/2) q[45];
r(0.998852062525034,pi/2) q[46];
r(0.392879479242762,pi/2) q[47];
r(0.0130796378757255,pi/2) q[48];
r(0.95032073655763,pi/2) q[49];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[38],q[39];
cx q[37],q[38];
cx q[36],q[37];
cx q[35],q[36];
cx q[34],q[35];
cx q[33],q[34];
cx q[32],q[33];
cx q[31],q[32];
cx q[30],q[31];
cx q[29],q[30];
cx q[28],q[29];
cx q[27],q[28];
cx q[26],q[27];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
r(0.448907473280915,pi/2) q[11];
r(0.839176893956702,pi/2) q[12];
r(0.689540150859248,pi/2) q[13];
r(0.223144565112061,pi/2) q[14];
r(0.031291327496725,pi/2) q[15];
r(0.186360526077401,pi/2) q[16];
r(0.539724001723092,pi/2) q[17];
r(0.925124074510095,pi/2) q[18];
r(0.393815270472374,pi/2) q[19];
r(0.49452624936307,pi/2) q[20];
r(0.469541236101464,pi/2) q[21];
r(0.304044034528234,pi/2) q[22];
r(0.145544589133136,pi/2) q[23];
r(0.955900947646189,pi/2) q[24];
r(0.0139832440667249,pi/2) q[25];
r(0.618747076723328,pi/2) q[26];
r(0.468442309293724,pi/2) q[27];
r(0.457429745692801,pi/2) q[28];
r(0.459892084851088,pi/2) q[29];
r(0.929535249468228,pi/2) q[30];
r(0.163054294679177,pi/2) q[31];
r(0.197638496990526,pi/2) q[32];
r(0.441134629473652,pi/2) q[33];
r(0.502958527404088,pi/2) q[34];
r(0.578788922678898,pi/2) q[35];
r(0.468359139511864,pi/2) q[36];
r(0.730359610592659,pi/2) q[37];
r(0.733503603324351,pi/2) q[38];
r(0.313676528444663,pi/2) q[39];
r(0.772384846979814,pi/2) q[40];
r(0.480047024185138,pi/2) q[41];
r(0.689462666173027,pi/2) q[42];
r(0.604757010574927,pi/2) q[43];
r(0.685838733430063,pi/2) q[44];
r(0.391732707016608,pi/2) q[45];
r(0.306017633341953,pi/2) q[46];
r(0.00567431862804146,pi/2) q[47];
r(0.91145775529201,pi/2) q[48];
r(0.916383492962683,pi/2) q[49];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[38],q[39];
cx q[37],q[38];
cx q[36],q[37];
cx q[35],q[36];
cx q[34],q[35];
cx q[33],q[34];
cx q[32],q[33];
cx q[31],q[32];
cx q[30],q[31];
cx q[29],q[30];
cx q[28],q[29];
cx q[27],q[28];
cx q[26],q[27];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
r(0.18086739088153,pi/2) q[12];
r(0.70899076358448,pi/2) q[13];
r(0.127778685795681,pi/2) q[14];
r(0.8307231450962,pi/2) q[15];
r(0.223348824194693,pi/2) q[16];
r(0.848321416827283,pi/2) q[17];
r(0.852053199384271,pi/2) q[18];
r(0.0193575213691345,pi/2) q[19];
r(0.757586912422426,pi/2) q[20];
r(0.623742286460689,pi/2) q[21];
r(0.589978122250624,pi/2) q[22];
r(0.191052594490398,pi/2) q[23];
r(0.779861836367368,pi/2) q[24];
r(0.118470451859792,pi/2) q[25];
r(0.826589772763862,pi/2) q[26];
r(0.482044613238078,pi/2) q[27];
r(0.87804564930718,pi/2) q[28];
r(0.0138176013369581,pi/2) q[29];
r(0.461901871885322,pi/2) q[30];
r(0.600665662946641,pi/2) q[31];
r(0.727001491414896,pi/2) q[32];
r(0.885127858425274,pi/2) q[33];
r(0.0813856563971314,pi/2) q[34];
r(0.359799318901298,pi/2) q[35];
r(0.414646689434621,pi/2) q[36];
r(0.993329257866914,pi/2) q[37];
r(0.177252307611507,pi/2) q[38];
r(0.06215276253215,pi/2) q[39];
r(0.245952766878675,pi/2) q[40];
r(0.557567699085341,pi/2) q[41];
r(0.184346252772636,pi/2) q[42];
r(0.619271455952228,pi/2) q[43];
r(0.00203459976964593,pi/2) q[44];
r(0.935867713067246,pi/2) q[45];
r(0.320806211289539,pi/2) q[46];
r(0.322791248254843,pi/2) q[47];
r(0.929861006798214,pi/2) q[48];
r(0.523656808894894,pi/2) q[49];
cx q[9],q[10];
r(0.680319648527479,pi/2) q[10];
cx q[10],q[11];
r(0.0821375232346133,pi/2) q[11];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
r(0.152265486519413,pi/2) q[0];
r(0.577708646422657,pi/2) q[1];
r(0.219198451097598,pi/2) q[2];
r(0.982252382727942,pi/2) q[3];
r(0.561300118416226,pi/2) q[4];
r(0.519973212737124,pi/2) q[5];
r(0.996627814865534,pi/2) q[6];
r(0.305997713289644,pi/2) q[7];
r(0.595632467377963,pi/2) q[8];
r(0.955646031001958,pi/2) q[9];
cx q[9],q[10];
r(0.0868815902935594,pi/2) q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
r(0.539376512889815,pi/2) q[0];
r(0.903893872450734,pi/2) q[1];
r(0.336150740943731,pi/2) q[2];
r(0.417487962334413,pi/2) q[3];
r(0.913728803403536,pi/2) q[4];
r(0.907189850829468,pi/2) q[5];
r(0.791741263428438,pi/2) q[6];
r(0.179745208906946,pi/2) q[7];
r(0.353602862425623,pi/2) q[8];
r(0.734455333234363,pi/2) q[9];
measure q[0] -> c36[0];
measure q[1] -> c36[1];
measure q[2] -> c36[2];
measure q[3] -> c36[3];
measure q[4] -> c36[4];
measure q[5] -> c36[5];
measure q[6] -> c36[6];
measure q[7] -> c36[7];
measure q[8] -> c36[8];
measure q[9] -> c36[9];
measure q[10] -> c36[10];
measure q[11] -> c36[11];
measure q[12] -> c36[12];
measure q[13] -> c36[13];
measure q[14] -> c36[14];
measure q[15] -> c36[15];
measure q[16] -> c36[16];
measure q[17] -> c36[17];
measure q[18] -> c36[18];
measure q[19] -> c36[19];
measure q[20] -> c36[20];
measure q[21] -> c36[21];
measure q[22] -> c36[22];
measure q[23] -> c36[23];
measure q[24] -> c36[24];
measure q[25] -> c36[25];
measure q[26] -> c36[26];
measure q[27] -> c36[27];
measure q[28] -> c36[28];
measure q[29] -> c36[29];
measure q[30] -> c36[30];
measure q[31] -> c36[31];
measure q[32] -> c36[32];
measure q[33] -> c36[33];
measure q[34] -> c36[34];
measure q[35] -> c36[35];
measure q[36] -> c36[36];
measure q[37] -> c36[37];
measure q[38] -> c36[38];
measure q[39] -> c36[39];
measure q[40] -> c36[40];
measure q[41] -> c36[41];
measure q[42] -> c36[42];
measure q[43] -> c36[43];
measure q[44] -> c36[44];
measure q[45] -> c36[45];
measure q[46] -> c36[46];
measure q[47] -> c36[47];
measure q[48] -> c36[48];
measure q[49] -> c36[49];
