import json
from datetime import datetime
from qiskit import IBMQ

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
        
        
provider = IBMQ.load_account()
backends = provider.backends()
backend = provider.get_backend("ibm_lagos")

for i in range(1, 12):
    for j in range(1, 28):
        t = datetime(day=j, month=i, year=2022, hour=10)

        properties = backend.properties(datetime=t)
        
        if properties is None:
            continue
            
        properties = properties.to_dict()

        convert_dict_to_json(properties, "callibration_data/ibm_lagos" + datetime_to_str(t) + ".json")