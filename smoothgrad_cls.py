import torch
from typing import List
from torch.utils.hooks import RemovableHandle
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor

class SmoothGradient:
    def __init__(self, stdev, num_samples, model):
        self.stdev = stdev
        self.num_samples = num_samples
        #self.num_samples = 1
        self.model = model
        self.attention_weight = None

    def forward_atten_hook(self, module, input, output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scale = output[0].detach().max() - output[0].detach().min()
        noise = torch.randn(output[0].shape, device=device) * self.stdev * scale
        output = (torch.add(output[0],noise),)
        return output

    def forward_atten_hook1(self, module, input, output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scale = output.detach().max() - output.detach().min()
        noise = torch.randn(output.shape, device=device) * self.stdev * scale
        output += noise
        return output

    def register_attention_gradient_hooks(self,weight_list, attention_grad,index=None):  #index
        hooks = []

        def backward_hook(module, grad_in, grad_out):
            # print("attention backward grad out: ",grad_out[0].shape)
            attention_grad.append(grad_out[0][0][:self.end_idx].detach())

        # hooks.append(self.attention_weight.register_backward_hook(backward_hook))
        hooks.append(weight_list[index].register_backward_hook(backward_hook))
        return hooks

    def get_gradients(self, model_input, target):
        original_requires_grad_dict = {}
        self.end_idx = len(model_input[0]) if model_input[0][-1]!=0 else ((model_input[0] == 0).nonzero(as_tuple=True)[0].tolist()[0])  #no [PAD] after case, end at len
        for param_name, param in self.model.named_parameters():
            original_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        hooks_list1 = {}
        hooks_list2 = {}
        hooks_list3 = {}
        attention_grad_list1 = [[] for _ in range(len(self.attention_weight_list1))]
        attention_grad_list2 = [[] for _ in range(len(self.attention_weight_list2))]
        attention_grad_list3 = [[] for _ in range(len(self.attention_weight_list3))]
        for i in range(len(self.attention_weight_list1)):
            hooks_list1[i]=self.register_attention_gradient_hooks(self.attention_weight_list1,attention_grad_list1[i],i)
            hooks_list2[i] = self.register_attention_gradient_hooks(self.attention_weight_list2,attention_grad_list2[i], i)
            hooks_list3[i] = self.register_attention_gradient_hooks(self.attention_weight_list3,attention_grad_list3[i], i)


        output = self.model(
            input_ids=model_input[0].unsqueeze(0).to('cpu'),
            token_type_ids=model_input[1].unsqueeze(0).to('cpu'),
            attention_mask=model_input[2].unsqueeze(0).to('cpu'),
            labels=torch.tensor(target).unsqueeze(0).to('cpu')
        )
        loss = output.loss
        loss.backward()

        ###################
        for key, value in hooks_list1.items():
            for hook in value:
                hook.remove()
        for key, value in hooks_list2.items():
            for hook in value:
                hook.remove()
        for key, value in hooks_list3.items():
            for hook in value:
                hook.remove()

        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_requires_grad_dict[param_name]
        # return attention_grad
        return attention_grad_list1,attention_grad_list2,attention_grad_list3
        # return embedding_grad,attention_grad_list     #[0].shape   [10,768]

    def smooth_grad(self, model_input, target,normalize=True):
        total_grads1, total_grads2,total_grads3 = None,None,None
        attention_grad1, attention_grad2,attention_grad3 = None,None,None
        for _ in range(self.num_samples):
            self.attention_weight_list1 = [lay.attention.self for lay in self.model.bert.encoder.layer]
            self.attention_weight_list2 = [lay.attention for lay in self.model.bert.encoder.layer]
            self.attention_weight_list3 = [lay.output for lay in self.model.bert.encoder.layer]

            handle_atten_list1 = {}
            handle_atten_list2 = {}
            handle_atten_list3 = {}
            for i in range(len(self.attention_weight_list3)):
                handle_atten_list1[i] = self.attention_weight_list1[i].register_forward_hook(self.forward_atten_hook)
                handle_atten_list2[i] = self.attention_weight_list2[i].register_forward_hook(self.forward_atten_hook)
                handle_atten_list3[i] = self.attention_weight_list3[i].register_forward_hook(self.forward_atten_hook1)


            try:
                grads_list1,grads_list2,grads_list3 = self.get_gradients(model_input, target)
                grads_list1,grads_list2,grads_list3 = torch.stack([grads[0].detach().cpu() for grads in grads_list1]),\
                              torch.stack([grads[0].detach().cpu() for grads in grads_list2]),\
                              torch.stack([grads[0].detach().cpu() for grads in grads_list3])


            # handle_atten.remove()
            finally:
                for key, value in handle_atten_list1.items():
                    value.remove()
                for key, value in handle_atten_list2.items():
                    value.remove()
                for key, value in handle_atten_list3.items():
                    value.remove()
            if total_grads1 is None:
                total_grads1 = grads_list1
            else:
                total_grads1 += grads_list1
            if total_grads2 is None:
                total_grads2 = grads_list2
            else:
                total_grads2 += grads_list2
            if total_grads3 is None:
                total_grads3 = grads_list3
            else:
                total_grads3 += grads_list3

        total_grads1 /= self.num_samples
        total_grads2 /= self.num_samples
        total_grads3 /= self.num_samples

        # normalize results
        if normalize:
            attention_grad1 = torch.sum(total_grads1, dim=2)
            attention_grad2 = torch.sum(total_grads2, dim=2)
            attention_grad3 = torch.sum(total_grads3, dim=2)

            # abs_embedding_grad = torch.abs(embedding_grad)
            # sigmoid_layer = torch.nn.Sigmoid()
            # total_grads = sigmoid_layer(embedding_grad)
        return attention_grad1,attention_grad2,attention_grad3