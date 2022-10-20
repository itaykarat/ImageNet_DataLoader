from torch import nn
import torchvision.models as models
from torch.nn import Identity


class run_model():
    def __init__(self):
        pass



    class Identity(nn.Module):  # Define the last layer as a [INPUT = OUTPUT] neurons for feature embedding.
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x


    def RUN_RESNET18(self,features):
        model = models.resnet18(pretrained=False)
        model.fc = Identity()
        x = features
        output = model(x)
        print(f'\n\nThis is the outpus of RESNET18: \n {output}')
        print(f'\n\nShape of the output: \n{output.shape}')
        return output

