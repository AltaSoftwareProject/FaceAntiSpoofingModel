from Model import DeePixBiS
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.jit.mobile import _backport_for_mobile,_get_model_bytecode_version
if __name__ == "__main__":
    model = DeePixBiS()
    model.load_state_dict(torch.load('./gpumodel_2.pth'))
    model.eval()
    model = torch.quantization.convert(model)
    scripted_model = torch.jit.script(model)
    opt_model = optimize_for_mobile(scripted_model)
    opt_model._save_for_lite_interpreter("gpumodel_2.ptl")
    print("test")
    

    # MODEL_INPUT_FILE = "mobile_model_121.pt"
    # MODEL_OUTPUT_FILE = "mobile_model_121_v7.ptl"

    # print("model version", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

    # _backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=7)

    # print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE))