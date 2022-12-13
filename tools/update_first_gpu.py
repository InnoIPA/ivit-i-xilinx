#!/usr/bin/python3
import json, os, sys, argparse, GPUtil
sys.path.append(os.getcwd())

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    ret = dict()
    for gpu in gpus:
        ret.update({ gpu.name: {
                
                "id": gpu.id,
                "name": gpu.name, 
                "uuid": gpu.uuid, 
                "load": round(gpu.load*100, 3), 
                "memoryUtil": round(gpu.memoryUtil*100, 3), 
                "temperature": gpu.temperature
        }})
    return ret

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--framework", default="tensorrt", help="framework [ tensorrt, openvino ]")
parser.add_argument("-j", "--json", help="path to task configuration")
args = parser.parse_args()

print("# Modify GPU Information to each configuration \n")

gpu_list = get_gpu_info()

task_cfg, mode_cfg_path, model_cfg = None, None, None

task_cfg_path = args.json
framework = args.framework

with open(args.json, 'r') as f:
    task_cfg = json.load(f)
    mode_cfg_path = task_cfg["prim"]["model_json"]

if None in [ task_cfg, mode_cfg_path]:
    raise Exception("Parse json failed ... ")

with open(mode_cfg_path, "r") as f:
    model_cfg = json.load(f)
    first_gpu = list(gpu_list.keys())[0]
    print("Detected GPU: {}".format(first_gpu))

    model_cfg[framework]["device"] = first_gpu

if None in [ model_cfg ]:
    raise Exception("Parse model json failed ... ")

with open(mode_cfg_path, "w") as f:
    json.dump(model_cfg, f, ensure_ascii=False, indent=4)