import torch
from Model import DeePixBiS
from torch.utils.mobile_optimizer import optimize_for_mobile

def ConvertModel():
    model = DeePixBiS()
    model.load_state_dict(torch.load('./MyModel.pth'))
    model.eval()
    model = torch.quantization.convert(model)

    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter("DeepPixBis_121_v5.ptl")

    print("Model successfully exported")

if __name__ == "__main__":
    main_models = ConvertModel()