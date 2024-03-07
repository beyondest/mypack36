from autoaim_alpha.autoaim_alpha.utils_network.actions import *
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *


weight_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/weights/11multi.pth'
output_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/weights/11multi.onnx'
dummy_input = torch.randn(1, 1,32,32)
model = QNet(11)

Data.save_model_to_onnx(model,
                        output_path,
                        dummy_input,
                        weight_path,
                        input_names=['input'],
                        output_names=['output'],
                        if_dynamic_batch_size=True
                        )
Data.save_model_to_onnx()


