from sigopt.examples import  franke_function
from comet_ml import Optimizer 
import sys

def fit(x,y):
    return franke_function(x,y)
    
   


opt=Optimizer(sys.argv[1],api_key="6krXLYdn4mMFKPsF8jwrFwXtu",workspace="returncode13",project_name="franke-function-opt-distributed-01")

for experiment in opt.get_experiments():
    val=fit(experiment.get_parameter('x'),experiment.get_parameter('y'))
    experiment.log_metric("val",val)
    #print(experiment.get_parameter('x'),experiment.get_parameter('y'))