from configobj import ConfigObj
from glob import glob
import os
import subprocess

from tqdm import tqdm
EXPPATH = "/experiment/code-2019-nalu/experiments/params/configs"


if not os.path.exists(EXPPATH):
    os.makedirs(EXPPATH)

f = 0
for seed in range(100, 120):
    for optim in ["rmsprop"]: #"adam",
        for operation in ["MUL", "ADD", "SUB", "DIV"]:
            for arch in ["NALU"]:
                for nalu in ["NALU"]:
                    for mg in ["-1.0", "0.0", "1.0"]:
                        for mw in ["-1.0", "0.0", "1.0"]:
                            for mm in ["-1.0", "0.0", "1.0"]:
                                for std in ["0.1", "0.5",]:
                                    if arch != "NALU":
                                        activations = ["relu"] # "softexp",
                                    else:
                                        activations = ["none"]  # nalu has no activation layers

                                    for activ in activations:
                                        f += 1
                                        config = ConfigObj()
                                        config.filename = EXPPATH+"/%s.ini" % "_".join([arch, nalu, activ, operation, str(seed), str(mg), str(mw), str(mm), str(std)])
                                        config["DEFAULT"] = {}
                                        config["DEFAULT"]["name"] = "nalu_org"      # --output
                                        config["DEFAULT"]["seed"] = str(seed)            # --seed
                                        config["DEFAULT"]["batchsize"] = "64"       # --batch
                                        config["DEFAULT"]["maxepoch"] = "201"       # --epochs
                                        config["DEFAULT"]["gpu"] = "-1"             # --gpu
                                        config["DEFAULT"]["learningrate"] = "0.01"  # --learningrate
                                        config["DEFAULT"]["optimizer"] = optim     # --optimizer
                                        config["DEFAULT"]["operation"] = operation      # --operation
                                        config["DEFAULT"]["PID"] = "0"
                                        config["DEFAULT"]["SUCCESS"] = ""
                                        config["DEFAULT"]["OVERWRITELOCK"] = "False"

                                        config["DATASET"] = {}
                                        config["DATASET"]["INTERP"] = {}
                                        config["DATASET"]["INTERP"]["dist"] = "gauss"
                                        config["DATASET"]["INTERP"]["mean"] = "0"
                                        config["DATASET"]["INTERP"]["stdv"] = "1"
                                        config["DATASET"]["INTERP"]["min"] = "-2"
                                        config["DATASET"]["INTERP"]["max"] = "2"
                                        config["DATASET"]["EXTRAP"] = {}
                                        config["DATASET"]["EXTRAP"]["dist"] = "gauss"
                                        config["DATASET"]["EXTRAP"]["mean"] = "10"
                                        config["DATASET"]["EXTRAP"]["stdv"] = "1"
                                        config["DATASET"]["EXTRAP"]["lt"] = "-2"
                                        config["DATASET"]["EXTRAP"]["gt"] = "2"

                                        config["ARCHITECTURE"] = {}
                                        config["ARCHITECTURE"]["hidden1"] = "2"                 # --hidden1
                                        config["ARCHITECTURE"]["hidden2"] = "1"                 # --hidden2
                                        config["ARCHITECTURE"]["architecture"] = arch           # --run
                                        config["ARCHITECTURE"]["activation"] = activ            # --activation
                                        config["ARCHITECTURE"]["nalu"] = nalu                   # --nalu
                                        config["ARCHITECTURE"]["loss"] = "rsme"                 # --loss
                                        config["ARCHITECTURE"]["weightclipping"] = "True"       # --weightc
                                        config["ARCHITECTURE"]["layerwisetraining"] = "False"   # --layerwisetraining

                                        config["ARCHITECTURE"]["INIT"] = {}
                                        config["ARCHITECTURE"]["INIT"]["G"] = {}
                                        config["ARCHITECTURE"]["INIT"]["G"]["mean"] = mg     # --ig
                                        config["ARCHITECTURE"]["INIT"]["G"]["stdv"] = std     # --isg
                                        config["ARCHITECTURE"]["INIT"]["W"] = {}
                                        config["ARCHITECTURE"]["INIT"]["W"]["mean"] = mw    # --iw
                                        config["ARCHITECTURE"]["INIT"]["W"]["stdv"] = std    # --isw
                                        config["ARCHITECTURE"]["INIT"]["M"] = {}
                                        config["ARCHITECTURE"]["INIT"]["M"]["mean"] = mm    # --im
                                        config["ARCHITECTURE"]["INIT"]["M"]["stdv"] = std     # --ism

                                        config.write()
                                        print("Experiment config Written")


print(f, "runs written")