from utils import *
from models import *

modeltype = 'SkyNet()'
weightfile = 'dac.weights'

model = eval(modeltype)
load_net(weightfile, model)
model = model.cuda()
model.eval()

print("Ready to run.")
