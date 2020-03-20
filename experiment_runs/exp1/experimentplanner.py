from configobj import ConfigObj
from glob import glob
import os
import subprocess

from tqdm import tqdm

TAG = "exp1"
EXPPATH = "/experiment/code-2019-nalu/experiments/" + TAG
CONFIGPATH = EXPPATH + "/configs"
REPO = "code-2019-nalu"
SCRIPT = "nalu_syn_simple_arith.py"
REPO_URL = "" # Add repo here

def main():

    if not os.path.exists(CONFIGPATH):
        os.makedirs(CONFIGPATH)

    name = "nalu_minimal_arith"

    paramsd = {
        "normal": ["(-3,3)", "(-2,4)", "(-4,2)", "(2,8)", "(-8,-2)"
                   ],
        "uniform": ["(0,1)", "(-1,0)", "(-1,1)", "(0,5)", "(-5,0)", "(-5,5)",
                    "(2,3)", "(-3,-2)"],
        "exponential": ["0.2", "0.4", "0.6", "0.8", "1"]
    }
    extd = {
        "normal": ["(8,10)"],
        "uniform": ["(-10, -5)"],
        "exponential": ["0.5"]
    }

    f = 0
    for seed in range(100, 110):
        for op in ["add","sub","mul","div"]:
            for dist in ["uniform", "normal", "exponential"]:
                for params in paramsd[dist]:
                    for ext in extd[dist]:
                        for nalu in ["nalu__paperm","nalu__paperv","nalui1", "nalui2"]:
                            f += 1
                            config = ConfigObj()
                            config.filename = CONFIGPATH+"/%s.ini" % "_".join([name, nalu, dist, params, ext, str(seed), op])
                            config["DEFAULT"] = {}
                            config["DEFAULT"]["name"] = name
                            config["DEFAULT"]["seed"] = seed
                            config["DEFAULT"]["nalu"] = nalu
                            config["DEFAULT"]["dist"] = dist
                            config["DEFAULT"]["op"] = op
                            config["DEFAULT"]["params"] = params
                            config["DEFAULT"]["ext"] = ext
                            config["DEFAULT"]["PID"] = "0"
                            config["DEFAULT"]["SUCCESS"] = ""
                            config["DEFAULT"]["OVERWRITELOCK"] = "False"
                            config.write()
                            print("Experiment config Written to " + CONFIGPATH)


    print(f, "runs written")




    if REPO_URL != "":
        if not os.path.exists(EXPPATH + "/" + REPO):
            gitclone = ["clone", "-b", TAG, "--single-branch", REPO_URL]

            proc = subprocess.Popen(["git"] + gitclone, shell=False, cwd=EXPPATH)

            exit = proc.wait()

            if exit != 0:
                raise Exception("Git failed!")
        else:
            print("Using existing repository.")



if __name__ == "__main__":
    main()
