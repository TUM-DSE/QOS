from qiskit import QuantumCircuit, transpile, IBMQ, Aer
from qiskit.quantum_info import hellinger_fidelity
import os
import shutil
from datetime import datetime
from qiskit.providers.fake_provider import *
from qiskit.providers.ibmq import AccountProvider
import matplotlib.pyplot as plt
from benchmarks import *
from collections import Counter
import mapomatic as mm
import json
import random
from types import MethodType
from qiskit.providers.ibmq.managed import IBMQJobManager
from matplotlib.colors import CSS4_COLORS as fancy_colors
from qiskit.providers.models import BackendConfiguration
from qiskit_aer.noise import NoiseModel
from benchmarks._utils import _get_ideal_counts
#from ._utils import perfect_counts, fidelity
from qiskit_aer.noise import NoiseModel

# from ._utils import perfect_counts, fidelity

def datetime_to_str(obj):
    """Helper function to convert datetime objects to strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj)} not serializable")

def convert_dict_to_json(d, file_path):
    """Recursively convert a dictionary with datetime objects to a JSON file"""
    # Recursively convert nested dictionaries
    for key, value in d.items():
        if isinstance(value, dict):
            convert_dict_to_json(value, file_path)

    # Convert datetime objects to strings using helper function
    d_str = json.dumps(d, default=datetime_to_str, indent=4)

    # Write JSON string to file
    with open(file_path, 'w') as f:
        f.write(d_str)
        
def get_callibration_data(machinename):  
    provider = IBMQ.load_account()
    backends = provider.backends()
    backend = provider.get_backend(machinename)
    
    ranges = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    
    for i,m in enumerate(ranges):

        for j in range(1, m+1):
            t = datetime(day=j, month=i+1, year=2022, hour=10)

            properties = backend.properties(datetime=t)
            
            if properties is None:
                continue
                
            properties = properties.to_dict()

            convert_dict_to_json(properties, "callibration_data/" + machinename + datetime_to_str(t) + ".json")

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    # print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)


def iterate_files_in_directory():
    #provider = IBMQ.load_account()
    benches = [GHZBenchmark(5), VanillaQAOABenchmark(5), FermionicSwapQAOABenchmark(5), HamiltonianSimulationBenchmark(5), BitCodeBenchmark(3), PhaseCodeBenchmark(3), MerminBellBenchmark(5)]
    qcs = [bench.circuit() for bench in benches]
    
    #ranges = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ranges = [31, 31, 30, 31, 30, 31]
    
    backend_names = ["ibm_perth", "ibm_lagos", "ibm_nairobi", "ibm_oslo", "ibmq_jakarta", "ibmq_manila", "ibmq_quito", "ibmq_belem", "ibmq_lima"]
    #backend_names = ["ibm_lagos", "ibm_nairobi", "ibmq_jakarta"]
    city_names = ["perth", "lagos", "nairobi", "oslo", "jakarta", "jakarta", "manila", "quito", "belem", "lima"]
    #city_names = ["lagos", "nairobi", "jakarta"]
    fake_backend_names = ["FakePerth", "FakeLagosV2", "FakeNairobiV2", "FakeOslo", "FakeJakartaV2", "FakeManilaV2", "FakeQuitoV2", "FakeBelemV2", "FakeLimaV2"]
    #fake_backend_names = ["FakeLagos", "FakeNairobi", "FakeJakarta"]
    backends = [eval(b)() for b in fake_backend_names]
    backend = None

    
    for i,m in enumerate(ranges):
        for j in range(1, m+1):
            t = datetime(day=j, month=i+1, year=2022, hour=10)

            date = datetime_to_str(t)
            fids = []
            
            for l,qc in enumerate(qcs):                
                best_fid = 0
                best_backend = ""
                
                for k, b in enumerate(backend_names):
                    filename = b + date + ".json"
                    file_path = os.path.join("callibration_data/" + b, filename)

                    if os.path.isfile(file_path):
                        move_and_rename_file(file_path,
                        "/mnt/c/Users/giort/Documents/GitHub/ME/lib/python3.9/site-packages/qiskit/providers/fake_provider/backends/" + city_names[k],
                        "props_" + city_names[k] + ".json")
                        
                    #backends.append(eval(fake_backend_names[k])())
                    backend = backends[k]
                    """
                    trans_qcs = [transpile(qc, backends[0], optimization_level=3) for qc in qcs]
                    #print(trans_qcs)
                    small_qcs = [mm.deflate_circuit(trans_qc) for trans_qc in trans_qcs]
                    
                    bests = [mm.best_overall_layout(small_qc, backends) for small_qc in small_qcs]
                    
                    machines = [best[1] for best in bests]
                    """
                    #fids.append([])
                    
                    trans_qc = transpile(qc, backend)

                    result = backend.run(trans_qc, shots=8192).result()
                    counts = result.get_counts()
                    
                    counts_copy = {}

                    for (k, v) in counts.items():
                        counts_copy[k.replace(" ", "")] = counts[k]
                        
                    
                    fid = benches[l].score(Counter(counts_copy))

                    
                    if fid > best_fid:
                        best_fid = fid
                        best_backend = b
                    
                fids.append(best_backend)
                    
                   
            print(fids)
                
            fids.clear()       
            #backends.clear()
    

def all_benchs_vs_all_backends():
    benches = [GHZBenchmark(7), VanillaQAOABenchmark(7), FermionicSwapQAOABenchmark(7), HamiltonianSimulationBenchmark(7), BitCodeBenchmark(4), PhaseCodeBenchmark(4), MerminBellBenchmark(7), VQEBenchmark(3)]
    qcs = [bench.circuit() for bench in benches]
    
    backends = FakeProviderForBackendV2().backends()
    #del backends[1]
    
    fids = {}
    
    for i,qc in enumerate(qcs):
        name = benches[i].name()
        fids[name] = {}
        print(name)
        for j,b in enumerate(backends):
            #print(b.backend_name)
            
            internal_fids = []
            for k in range(5):
                try:
                    trans_qc = transpile(qc, b, optimization_level=3)
                except:
                    break
     
                result = b.run(trans_qc, shots=8192).result()
                counts = result.get_counts()
                
                if isinstance(counts, dict):
                
                    counts_copy = {}
                    
                    for (k, v) in counts.items():
                        counts_copy[k.replace(" ", "")] = counts[k]
                        
                    counts = counts_copy
                    counts = Counter(counts)
                else:
                    counts[0] = Counter(counts[0])
                    counts[1] = Counter(counts[1])
                
                
                internal_fids.append(benches[i].score(counts))
            
            if len(internal_fids) > 0:
                internal_fids.sort()
                fids[name][b.backend_name] = internal_fids[2]
            
        fids[name] = sorted(fids[name].items(), key=lambda x:x[1], reverse=True)
        
    print(fids)
    
    
#def get_globally_optimal(fids):
    
    
def show_benefit():
    fids = {'GHZ': [('fake_oslo', 0.1143303231997449), ('fake_rochester_v2', 0.39221790708358334), ('fake_yorktown_v2', 0.5203179850071499), ('fake_cambridge_v2', 0.5611937548119268), ('fake_johannesburg_v2', 0.6644505068885331), ('fake_singapore_v2', 0.7000034388508504), ('fake_london_v2', 0.7136192478989355), ('fake_melbourne_v2', 0.7204424479899758), ('fake_almaden_v2', 0.7401007124783688), ('fake_bogota_v2', 0.7405880522595071), ('fake_burlington_v2', 0.7534546435066841), ('fake_essex_v2', 0.7552740221548059), ('fake_toronto_v2', 0.7872145938127164), ('fake_boeblingen_v2', 0.7896973407989226), ('fake_valencia_v2', 0.7928895262316332), ('fake_poughkeepsie_v2', 0.8012514030151613), ('fake_belem_v2', 0.8035469033185177), ('fake_lima', 0.8052033121633442), ('fake_cairo_v2', 0.8073683220212232), ('fake_manila_v2', 0.8083432225700297), ('fake_vigo_v2', 0.8206329206589391), ('fake_nairobi_v2', 0.8258921425598238), ('fake_ourense_v2', 0.8413638708332206), ('fake_geneva', 0.8494749860899901), ('fake_mumbai_v2', 0.849794935898974), ('fake_rome_v2', 0.8500121738666438), ('fake_quito_v2', 0.8589070155224126), ('fake_brooklyn_v2', 0.8597858112346891), ('fake_perth', 0.8681001905859013), ('fake_casablanca_v2', 0.8709225815590302), ('fake_manhattan_v2', 0.8733678339792315), ('fake_athens_v2', 0.8807828141223591), ('fake_paris_v2', 0.8866432385357669), ('fake_guadalupe_v2', 0.8877490542984056), ('fake_sydney_v2', 0.8900262060083691), ('fake_auckland', 0.9148449617748696), ('fake_montreal_v2', 0.914926863259179), ('fake_santiago_v2', 0.9149658592444216), ('fake_lagos_v2', 0.9153310998234903), ('fake_washington_v2', 0.9354120430515751), ('fake_hanoi_v2', 0.9354243228097947), ('fake_sherbrooke', 0.937795943348312), ('fake_prague', 0.9482206738479177), ('fake_kolkata_v2', 0.9499135077216475)], 'QAOA': [('fake_oslo', 0.5013113221728664), ('fake_rochester_v2', 0.6691605602997754), ('fake_bogota_v2', 0.6796511376827072), ('fake_yorktown_v2', 0.7130898530908023), ('fake_johannesburg_v2', 0.7163681585229684), ('fake_cambridge_v2', 0.7185974062168415), ('fake_toronto_v2', 0.740955449264215), ('fake_poughkeepsie_v2', 0.745020548000101), ('fake_melbourne_v2', 0.7621333023560085), ('fake_essex_v2', 0.7760333173883931), ('fake_almaden_v2', 0.7776069039958329), ('fake_singapore_v2', 0.7970800382629001), ('fake_burlington_v2', 0.7989158893049131), ('fake_london_v2', 0.8090786361446284), ('fake_boeblingen_v2', 0.8280272415425489), ('fake_perth', 0.8293385637154154), ('fake_athens_v2', 0.8329446996907982), ('fake_belem_v2', 0.8339937574290914), ('fake_manhattan_v2', 0.8386489511427674), ('fake_valencia_v2', 0.8425829176613668), ('fake_lima', 0.8480249046787627), ('fake_rome_v2', 0.8484183013306226), ('fake_manila_v2', 0.8558272716073182), ('fake_geneva', 0.8580565193011912), ('fake_vigo_v2', 0.8602202008864209), ('fake_nairobi_v2', 0.8692027577705562), ('fake_cairo_v2', 0.8751037075484553), ('fake_casablanca_v2', 0.875234839765742), ('fake_ourense_v2', 0.8766772941558951), ('fake_brooklyn_v2', 0.8791032401756982), ('fake_auckland', 0.8805456945658513), ('fake_guadalupe_v2', 0.8820537150646477), ('fake_paris_v2', 0.8886103259289801), ('fake_quito_v2', 0.8909707058401397), ('fake_lagos_v2', 0.8948391062500959), ('fake_mumbai_v2', 0.8984452422254787), ('fake_sydney_v2', 0.9053296836330277), ('fake_washington_v2', 0.9055919480676009), ('fake_montreal_v2', 0.9104438401072069), ('fake_kolkata_v2', 0.914115542191233), ('fake_santiago_v2', 0.914508938843093), ('fake_hanoi_v2', 0.9148367693863096), ('fake_prague', 0.9539141701377306)], 'FermionicQAOA': [('fake_oslo', 0.503001977916791), ('fake_rochester_v2', 0.6422079824582727), ('fake_toronto_v2', 0.6433230028273667), ('fake_yorktown_v2', 0.6633075986734327), ('fake_cambridge_v2', 0.6995028752701704), ('fake_johannesburg_v2', 0.7041344983417908), ('fake_melbourne_v2', 0.7520803742128248), ('fake_bogota_v2', 0.7529380821890509), ('fake_essex_v2', 0.7681195133682512), ('fake_london_v2', 0.7909345455358632), ('fake_poughkeepsie_v2', 0.7977962093456713), ('fake_singapore_v2', 0.8001977916791041), ('fake_almaden_v2', 0.8027709156077822), ('fake_perth', 0.8097183502152129), ('fake_burlington_v2', 0.8124630157391362), ('fake_manila_v2', 0.8277302177159591), ('fake_belem_v2', 0.8303033416446371), ('fake_lima', 0.8370792346568227), ('fake_valencia_v2', 0.8411962329427075), ('fake_boeblingen_v2', 0.8427401072999143), ('fake_athens_v2', 0.846256710002441), ('fake_vigo_v2', 0.8472859595739122), ('fake_rome_v2', 0.85183181184791), ('fake_casablanca_v2', 0.8652120562770358), ('fake_nairobi_v2', 0.8672705554199782), ('fake_ourense_v2', 0.8717306368963536), ('fake_manhattan_v2', 0.8755903227893705), ('fake_paris_v2', 0.8761049475751062), ('fake_guadalupe_v2', 0.878935383896652), ('fake_quito_v2', 0.8810796538372171), ('fake_geneva', 0.8875124636589121), ('fake_cairo_v2', 0.8894851920042319), ('fake_brooklyn_v2', 0.8943741274687202), ('fake_mumbai_v2', 0.9029512072309803), ('fake_sydney_v2', 0.904323539992942), ('fake_lagos_v2', 0.9098986418384111), ('fake_santiago_v2', 0.9113567453979953), ('fake_washington_v2', 0.9119571409813535), ('fake_montreal_v2', 0.9145302649100315), ('fake_auckland', 0.9242223650413854), ('fake_hanoi_v2', 0.9285966757201382), ('fake_sherbrooke', 0.9341717775656072), ('fake_kolkata_v2', 0.9386318590419824), ('fake_prague', 0.966678909864573)], 'Hamiltonian': [('fake_oslo', 0.639146804719227), ('fake_rochester_v2', 0.8377063750317271), ('fake_yorktown_v2', 0.8596790312817271), ('fake_cambridge_v2', 0.9200794219067271), ('fake_melbourne_v2', 0.9263538359692272), ('fake_essex_v2', 0.9290882109692271), ('fake_london_v2', 0.929454421906727), ('fake_almaden_v2', 0.9348743437817273), ('fake_johannesburg_v2', 0.9360462187817271), ('fake_singapore_v2', 0.940294265656727), ('fake_bogota_v2', 0.945421218781727), ('fake_lima', 0.9458850859692272), ('fake_boeblingen_v2', 0.9461292265942272), ('fake_valencia_v2', 0.9470569609692272), ('fake_burlington_v2', 0.9479358672192271), ('fake_manila_v2', 0.9485462187817271), ('fake_belem_v2', 0.9509387969067271), ('fake_poughkeepsie_v2', 0.9529651640942272), ('fake_nairobi_v2', 0.9554553984692271), ('fake_ourense_v2', 0.964342117219227), ('fake_quito_v2', 0.9678577422192272), ('fake_vigo_v2', 0.968858718781727), ('fake_rome_v2', 0.9700794219067271), ('fake_guadalupe_v2', 0.9707141875317271), ('fake_cairo_v2', 0.9710071562817271), ('fake_perth', 0.9711536406567272), ('fake_toronto_v2', 0.971422195344227), ('fake_casablanca_v2', 0.971837234406727), ('fake_sydney_v2', 0.972398757844227), ('fake_athens_v2', 0.9724475859692271), ('fake_brooklyn_v2', 0.9729846953442273), ('fake_mumbai_v2', 0.9734485625317271), ('fake_paris_v2', 0.9735706328442271), ('fake_geneva', 0.9768665312817271), ('fake_manhattan_v2', 0.9822620390942269), ('fake_santiago_v2', 0.9839221953442272), ('fake_auckland', 0.9841419219067271), ('fake_lagos_v2', 0.9851917265942272), ('fake_montreal_v2', 0.9862415312817271), ('fake_hanoi_v2', 0.9864856719067272), ('fake_washington_v2', 0.9889759062817272), ('fake_sherbrooke', 0.989000320344227), ('fake_prague', 0.9897327422192271), ('fake_kolkata_v2', 0.9913928984692271)], 'BitCode': [('fake_oslo', 0.10241699218750007), ('fake_cairo_v2', 0.712890625), ('fake_yorktown_v2', 0.7274169921875001), ('fake_bogota_v2', 0.7416992187500001), ('fake_geneva', 0.783203125), ('fake_mumbai_v2', 0.8316650390625001), ('fake_rome_v2', 0.8629150390624999), ('fake_perth', 0.8675537109375), ('fake_lima', 0.8792724609375), ('fake_manhattan_v2', 0.8802490234375), ('fake_belem_v2', 0.8836669921875001), ('fake_manila_v2', 0.8870849609374999), ('fake_nairobi_v2', 0.8936767578125), ('fake_sydney_v2', 0.9046630859375), ('fake_brooklyn_v2', 0.90673828125), ('fake_auckland', 0.9083251953125), ('fake_casablanca_v2', 0.9095458984375), ('fake_lagos_v2', 0.91162109375), ('fake_santiago_v2', 0.9185791015625001), ('fake_athens_v2', 0.9208984375000001), ('fake_prague', 0.9239501953125), ('fake_quito_v2', 0.925048828125), ('fake_hanoi_v2', 0.92529296875), ('fake_paris_v2', 0.9360351562500001), ('fake_guadalupe_v2', 0.9394531250000001), ('fake_washington_v2', 0.9410400390625), ('fake_montreal_v2', 0.9467773437499999), ('fake_toronto_v2', 0.9468994140625001), ('fake_kolkata_v2', 0.9555664062500001)], 'PhaseCode': [('fake_oslo', 0.054077148437500035), ('fake_yorktown_v2', 0.6430664062499999), ('fake_cairo_v2', 0.7045898437500001), ('fake_bogota_v2', 0.7408447265625), ('fake_geneva', 0.777099609375), ('fake_mumbai_v2', 0.823486328125), ('fake_lima', 0.8558349609375), ('fake_perth', 0.85693359375), ('fake_rome_v2', 0.8642578125), ('fake_belem_v2', 0.872802734375), ('fake_manila_v2', 0.8835449218749999), ('fake_manhattan_v2', 0.8841552734374999), ('fake_nairobi_v2', 0.8912353515625001), ('fake_sydney_v2', 0.8992919921875), ('fake_casablanca_v2', 0.9047851562499999), ('fake_lagos_v2', 0.9050292968749999), ('fake_brooklyn_v2', 0.9071044921875), ('fake_quito_v2', 0.9086914062499999), ('fake_auckland', 0.9090576171874999), ('fake_toronto_v2', 0.9101562500000001), ('fake_santiago_v2', 0.9165039062499999), ('fake_hanoi_v2', 0.9215087890625001), ('fake_athens_v2', 0.9218749999999999), ('fake_prague', 0.926025390625), ('fake_paris_v2', 0.9270019531249999), ('fake_washington_v2', 0.9342041015624999), ('fake_montreal_v2', 0.937744140625), ('fake_guadalupe_v2', 0.93798828125), ('fake_kolkata_v2', 0.9521484374999999)], 'MerminBell': [('fake_oslo', 0.50054931640625), ('fake_toronto_v2', 0.53253173828125), ('fake_yorktown_v2', 0.5421142578125), ('fake_bogota_v2', 0.5447998046875), ('fake_rochester_v2', 0.5535888671875), ('fake_poughkeepsie_v2', 0.57080078125), ('fake_johannesburg_v2', 0.57562255859375), ('fake_cambridge_v2', 0.616455078125), ('fake_essex_v2', 0.63287353515625), ('fake_melbourne_v2', 0.63671875), ('fake_burlington_v2', 0.653564453125), ('fake_athens_v2', 0.6572265625), ('fake_almaden_v2', 0.6610107421875), ('fake_belem_v2', 0.6912841796875), ('fake_london_v2', 0.69805908203125), ('fake_singapore_v2', 0.70416259765625), ('fake_boeblingen_v2', 0.70501708984375), ('fake_rome_v2', 0.708740234375), ('fake_lima', 0.7174072265625), ('fake_perth', 0.71923828125), ('fake_auckland', 0.72454833984375), ('fake_manila_v2', 0.7445068359375), ('fake_vigo_v2', 0.74481201171875), ('fake_valencia_v2', 0.74578857421875), ('fake_manhattan_v2', 0.748046875), ('fake_geneva', 0.75189208984375), ('fake_casablanca_v2', 0.7559814453125), ('fake_ourense_v2', 0.76141357421875), ('fake_nairobi_v2', 0.76605224609375), ('fake_cairo_v2', 0.7718505859375), ('fake_guadalupe_v2', 0.77294921875), ('fake_quito_v2', 0.77685546875), ('fake_brooklyn_v2', 0.7835693359375), ('fake_paris_v2', 0.79449462890625), ('fake_santiago_v2', 0.8009033203125), ('fake_lagos_v2', 0.804931640625), ('fake_washington_v2', 0.8094482421875), ('fake_sydney_v2', 0.8116455078125), ('fake_mumbai_v2', 0.82061767578125), ('fake_kolkata_v2', 0.8223876953125), ('fake_montreal_v2', 0.82623291015625), ('fake_hanoi_v2', 0.8349609375), ('fake_prague', 0.9168701171875)], 'VQE': [('fake_singapore_v2', 0.8532462775602603), ('fake_cambridge_v2', 0.8599519364248516), ('fake_rochester_v2', 0.8640634352907324), ('fake_london_v2', 0.8755168964171147), ('fake_johannesburg_v2', 0.9098772797962615), ('fake_poughkeepsie_v2', 0.9404687892150176), ('fake_washington_v2', 0.9406156284602276), ('fake_burlington_v2', 0.9451676450617386), ('fake_yorktown_v2', 0.9462934126083488), ('fake_rome_v2', 0.9509433220399997), ('fake_toronto_v2', 0.9554463922264406), ('fake_bogota_v2', 0.9607815514690716), ('fake_mumbai_v2', 0.9632778186376422), ('fake_geneva', 0.9636204435431323), ('fake_paris_v2', 0.9654804073157925), ('fake_kolkata_v2', 0.9663614427870527), ('fake_ourense_v2', 0.9667530141076128), ('fake_melbourne_v2', 0.9679766744843631), ('fake_quito_v2', 0.9700813703323735), ('fake_hanoi_v2', 0.9712560842940537), ('fake_auckland', 0.9731160480667141), ('fake_sherbrooke', 0.9763465114613348), ('fake_perth', 0.9765422971216148), ('fake_nairobi_v2', 0.9766891363668249), ('fake_oslo', 0.977423332592875), ('fake_casablanca_v2', 0.977668064668225), ('fake_prague', 0.9785980465545552), ('fake_manila_v2', 0.9787938322148353), ('fake_cairo_v2', 0.9793322427806054), ('fake_manhattan_v2', 0.9800174925915854), ('fake_athens_v2', 0.9806537959874956), ('fake_brooklyn_v2', 0.9813390457984758), ('fake_essex_v2', 0.9820242956094559), ('fake_santiago_v2', 0.982562706175226), ('fake_sydney_v2', 0.982562706175226), ('fake_lagos_v2', 0.983003223910856), ('fake_valencia_v2', 0.9834926880615562), ('fake_lima', 0.9847652948533764), ('fake_montreal_v2', 0.9849121340985865), ('fake_guadalupe_v2', 0.9851079197588665), ('fake_belem_v2', 0.9854015982492865), ('fake_vigo_v2', 0.9866742050411068), ('fake_boeblingen_v2', 0.9915199001330378), ('fake_almaden_v2', 0.99736926364607)]}
    
    #fids = {k: sorted(v, key=lambda x:x[1], reverse=True) for k,v in fids.items()}

    backends_used = {}
    
    fid_get_best = 0
    fid_get_random = 0
    fid_get_same = 0
    fid_get_optimal = 0
    
    for k,v in fids.items():
        fid_get_best = fid_get_best + v[0][1]
        
        fid_get_random = fid_get_random + v[random.randint(0, len(v))][1]
        
        v_as_dict = dict(v)
        
        fid_get_same = fid_get_same + v_as_dict['fake_kolkata_v2']
        
        for kk in v:
            if kk[0] in backends_used:
                continue
            backends_used[kk[0]] = kk[1]
            fid_get_optimal = fid_get_optimal + kk[1]
            break
            
    print(fid_get_best/8, fid_get_random/8, fid_get_same/8, fid_get_optimal/8)    
        #backends_used.clear()
    
    
def execute_on_backends():
    provider = FakeProviderForBackendV2()
    
    backends = provider.backends()
    
    #qubits = [i for i in range(2, 10)]
    qubits = [3, 6, 10]
       
    fids = {}

    for b in backends:
        for q in qubits:
            bench = HamiltonianSimulationBenchmark(q)
            qc = bench.circuit()
            name = b.backend_name + "_" + str(q)
            fids[name] = []
            try:
                cqc = transpile(qc, b)
            except:
                continue

            for i in range(5):
                result = b.run(cqc, shots=8192).result()
                counts = result.get_counts()
                # avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
                fids[name].append(bench.score(Counter(counts)))

            # print(fids)
            # fids.sort()
            #print(fids[b])
    print(fids)
    exit()
    means = [np.median(fids[name]) for name in fids]
    stds = [np.std(fids[name]) for name in fids]
    diff_all = np.max(np.abs(np.subtract.outer(means, means)))

    fig, ax = plt.subplots()
    ax.bar(
        fids.keys(),
        means,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.axhline(np.mean(means), color="green", linewidth=2)
    plt.text(
        0.01,
        0.95,
        "Max difference: {:.2f}".format(diff_all),
        transform=plt.gca().transAxes,
    )
    ax.set_xlabel("IBMQ Backend")
    ax.set_ylabel("Fidelity Score")
    plt.xticks(rotation="vertical")
    ax = plt.gca()
    ax.set_ylim([0.5, 1])
    # ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")


def run_all_benchs(dir_path):
    backend = FakeTorontoV2()

    for filename in os.listdir(dir_path):
        print(filename)
        file_path = os.path.join(dir_path, filename)
        qc = QuantumCircuit.from_qasm_file(file_path)
        perf_counts = perfect_counts(qc)
        qc = transpile(qc, backend, optimization_level=3)
        fids = []

        for i in range(5):
            job = backend.run(qc, shots=8192)

            counts = job.result().get_counts()

            fids.append(fidelity(perf_counts, counts))
        print(fids)
        fids.sort()
        print(fids[2])


def execute_benchmarks():
    benchmarks = [
        "HamiltonianSimulation",
        "VQE",
        "VanillaQAOA",
        "GHZ",
        "BitCode",
        "PhaseCode",
        "MerminBell",
        "FermionicSwapQAOA",
    ]
       
    
    fake_backend = FakeParisV2()
    
    qbits = [4, 6, 8, 10]
    #qbits = [2, 3]
    
    fids = {}
    circs = []   

    for q in qbits:
        key = str(q) + "q"
        fids[key] = {}

        for b in benchmarks:
            fids[key][b] = []
            #print(q, b)
            
            bench = eval(b + "Benchmark")(q)

            qc = bench.circuit()
            
            
            cqc = transpile(qc, fake_backend, optimization_level=3)
            
            #fid_buffer = []
            
            for i in range(5): 
                results = fake_backend.run(cqc, shots=8192).result()           
                counts = results.get_counts()
                counts_copy = {}                           
            
                if isinstance(counts, list):
                    fids[key][b].append(bench.score(counts))
                else:
                    for (k, v) in counts.items():
                        counts_copy[k.replace(" ", "")] = v
                    
                    fids[key][b].append(bench.score(Counter(counts_copy)))

            #fid_buffer.sort()
            #fids[key][b].append(fid_buffer[2])

    with open('scalability.txt', 'w') as file:
        file.write(json.dumps(fids))
    
def plot_results():
    with open('scalability2.txt') as f:
        data = f.read()
        
    fids = json.loads(data)
    # means = { q : [np.median(fids[b]) for b in fids[qbits]] for qbits in fids}
    means = {}
    for q, b in fids.items():
        means[q] = [np.median(fids[q][b]) for b in fids[q]]

    stds = {}
    for q, b in fids.items():
        stds[q] = [np.std(fids[q][b]) for b in fids[q]]
    # diff_all = np.max(np.abs(np.subtract.outer(means, means)))

    group_labels = [
        "Ham.Sim.",
        "VQE",
        "Van.QAOA",
        "GHZ",
        "BitCode",
        "PhaseCode",
        "MerminBell",
        "Ferm.QAOA",
    ]

    x = np.arange(len(group_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    
    hatches = [".", "-", "o", "|"]
    colors = [fancy_colors["lightsteelblue"], fancy_colors["bisque"], fancy_colors["turquoise"], fancy_colors["lightcoral"]]
    # fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots()
    
    fig.set_size_inches(13, 7)
    
    for i, (qbits, score) in enumerate(means.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=qbits, align='center', yerr=stds[qbits], hatch=hatches[i], color=colors[i])
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel("Benchmarks")
    ax.set_ylabel("Fidelity Score")
    #ax.set_xticks(x + 2 * width, group_labels, rotation=90)
    ax.set_xticks(x + 1.5 * width, group_labels)
    # Add title and legend
    # ax.set_title("Benchmark score with increasing number of qubits")
    lgnd = ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncols=4, handlelength=3, handleheight=2)

        
    plt.savefig('scalability.png', dpi=300, bbox_inches="tight")
    

def test_qasm_smilator():
    try:
        provider = IBMQ.load_account()
    except e:
        print(e)
        
    backend = provider.get_backend("ibmq_qasm_simulator")
    
    fake_backend = FakeTorontoV2()
    
    
    bench = VanillaQAOABenchmark(5)
    
    qc = bench.circuit()
    
    cqc = transpile(qc, fake_backend, optimization_level=3)
    
    noise_model = NoiseModel.from_backend(fake_backend)
    
    job = backend.run(cqc, shots=8192, noise_model=noise_model)
    
    results = job.result()
    counts = results.get_counts()
    
    #fid = hellinger_fidelity(perf_counts, counts)
    print(bench.score(Counter(counts)))

#test_qasm_smilator()
#execute_benchmarks()

def perfect_counts(original_circuit: QuantumCircuit):
    #provider = IBMQ.load_account()
    #backend = provider.get_backend("simulator_statevector")
    backend = Aer.get_backend('aer_simulator_statevector')
        
    cnt = (
        backend.run(original_circuit, shots=20000).result().get_counts()
    )
    #pdb.set_trace()
    return {k.replace(" ", ""): v for k, v in cnt.items()}
    
def test_counts():
    bench = BitCodeBenchmark(3)
    qc = bench.circuit()
    
    backend = FakeLagosV2()
    
    cqc = transpile(qc, backend, optimization_level=3)
    
    counts2 = Counter(perfect_counts(qc))
    
    job = backend.run(cqc, shots=8192)
    
    counts = Counter(job.result().get_counts())
    
    print(counts)
    print(counts2)
    
    #print(counts2)
    #perf_counts = {"00000" : 10000, "11111": 10000}
    #perf_counts2 = {"00000" : 4096, "11111": 4096}
    
    #fid1 = hellinger_fidelity(perf_counts2, counts)
    #fid2 = hellinger_fidelity(perf_counts, counts2)
    
    print(bench.score(counts))
    print(bench.score(counts2))

 
def run_scale():
    backend = FakeOslo()
    fids = {}
    #qubits = [i for i in range(2, 11, 2)]
    qubits = [i for i in range(2, 8)]
    #layers = [i for i in range(1, 11, 2)]
    layers = [3, 5, 7]
    
    for q in qubits:        
        for l in layers:
            bench = HamiltonianSimulationBenchmark(q, total_time=l)
            qc = bench.circuit()
            fids[str(q) + "_" + str(l)] = []
           
            cqc = transpile(qc, backend)

            for i in range(5):
                result = backend.run(cqc, shots=8192).result()
                #print(result)
                counts = result.get_counts()
                # avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
                fids[str(q) + "_" + str(l)].append(bench.score(Counter(counts)))

            # print(fids)
            # fids.sort()
        #print(fids[str(q) + "_" + str(l)])

    means = [np.median(fids[name]) for name in fids]
    stds = [np.std(fids[name]) for name in fids]
    diff_all = np.max(np.abs(np.subtract.outer(means, means)))

    fig, ax = plt.subplots()
    ax.bar(
        fids.keys(),
        means,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.axhline(np.mean(means), color="green", linewidth=2)
    plt.text(
        0.01,
        0.95,
        "Max difference: {:.2f}".format(diff_all),
        transform=plt.gca().transAxes,
    )
    ax.set_xlabel("IBMQ Backend")
    ax.set_ylabel("Fidelity Score")
    plt.xticks(rotation="vertical")
    ax = plt.gca()
    ax.set_ylim([0.5, 1])
    # ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")

    
 
#test_counts()
#execute_benchmarks()
execute_benchmarks()

