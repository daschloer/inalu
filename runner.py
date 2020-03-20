from configobj import ConfigObj
from glob import glob
import os
import subprocess
import experimentplanner


TAG = experimentplanner.TAG
EXPPATH = experimentplanner.EXPPATH
CONFIGPATH = experimentplanner.CONFIGPATH
REPO = experimentplanner.REPO
SCRIPT = experimentplanner.SCRIPT



conf2cli = {}
conf2cli["DEFAULT"] = {}
conf2cli["DEFAULT"]["name"] = "--output"
conf2cli["DEFAULT"]["seed"] = "--seed"
conf2cli["DEFAULT"]["nalu"] = "--nalu"
conf2cli["DEFAULT"]["dist"] = "--dist"
conf2cli["DEFAULT"]["params"] = "--params"
conf2cli["DEFAULT"]["ext"] = "--ext"
conf2cli["DEFAULT"]["op"] = "--operation"


OVERRIDE = False

def config2cli(cli_dict, config):
    ret = []
    for ckey in config:

        if ckey in cli_dict and type(cli_dict[ckey]) == str:
            ret.append(cli_dict[ckey])
            ret.append(config[ckey])
        elif ckey in cli_dict and type(cli_dict[ckey]) == dict:
            ret += config2cli(cli_dict[ckey], config[ckey])
    return ret

def createLock(filename):
    try:
        os.rename(filename,filename + ".lock")
        return True
    except FileNotFoundError:
        return False

configfiles = [f for f in glob(EXPPATH + "/configs/*.ini")]

import random
random.shuffle(configfiles)

print(len(configfiles), "config files to work on")
for configfile in configfiles:
    config = ConfigObj(configfile)

    if "DEFAULT" not in config:
        continue

    if not createLock(configfile):
        if OVERRIDE or "OVERWRITELOCK" in config["DEFAULT"] and config["DEFAULT"]["OVERWRITELOCK"] == "True":
            print("%s is already locked. Ignoring lock according to config instruction." % configfile)
        else:
            print("%s is already locked" % configfile)
            continue

    if config["DEFAULT"]["SUCCESS"] != "":
        continue

    cliargs = config2cli(conf2cli, config.dict())

    print(cliargs)
    print(" ".join(cliargs))

    proc = subprocess.Popen(["python3", SCRIPT] + cliargs, shell=False, cwd=EXPPATH + "/" + REPO)
    print(proc.pid)

    config.filename = config.filename + ".lock"
    config["DEFAULT"]["PID"] = str(proc.pid)
    config["DEFAULT"]["SUCCESS"] = ""

    config.write()

    exit = proc.wait()
    config["DEFAULT"]["SUCCESS"] = str(exit)
    config["DEFAULT"]["PID"] = "0"
    config.filename = config.filename + ".done"
    config.write()

    print("Process %i finished with exit code %i" %(proc.pid, exit))
    print(cliargs)
    print(" ".join(cliargs))

    os.remove(configfile+".lock")
    import sys
    sys.exit(exit)
    break # only one run per pod
