import torch 
from scipy.stats import qmc

def collocation_points(number_of_points,T,w_max,w_min):
  sampler=qmc.LatinHypercube(d=2)
  lower_bound=[0,w_min]
  upper_bound=[T,w_max]
  samples=sampler.random(number_of_points)
  scaled_samples=qmc.scale(samples,lower_bound,upper_bound)
  w=torch.tensor(scaled_samples[:,1],dtype=torch.float32).unsqueeze(1)
  t=torch.tensor(scaled_samples[:,0],dtype=torch.float32).unsqueeze(1)
  t.requires_grad_(True)
  w.requires_grad_(True)
  return w,t


def terminal_points(number_of_points,T,w_max,w_min,gamma):
  sampler = qmc.LatinHypercube(d=1)
  samples = sampler.random(number_of_points)
  scaled_samples = qmc.scale(samples,w_min,w_max)
  w_tc = torch.tensor(scaled_samples,dtype=torch.float32)
  t_tc = torch.ones(number_of_points,1) * T
  v_tc = (w_tc**(1-gamma))/(1-gamma)
  w_tc.requires_grad_(True)
  t_tc.requires_grad_(True)
  return w_tc,t_tc,v_tc
