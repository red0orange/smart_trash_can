# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午11:09
# @Author  : red0orange
# @File    : transformProto.py
# @Content : 把pytorch训练完成的模型转换为cv能够直接调用的proto文件
import torchvision
import torch


if __name__ == '__main__':
    model_path = "checkout_point/25.pth"
    save_path = "cls.proto"

    model = torchvision.models.squeezenet1_0(False)
    # model = torchvision.models.alexnet(pretrained=False).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dummy_input = torch.full((1, 3, 224, 224), 3)
    # print(model(dummy_input).argmax())
    torch.onnx.export(model, dummy_input, save_path, verbose=True)
