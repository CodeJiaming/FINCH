import torch
import torch.nn as nn
from torch.autograd import Variable


from nas_module.darts.operations import *
from nas_module.darts.genotypes import PRIMITIVES, Genotype
import random
import re
import sys
import copy



def softmax_with_mask(xs, mask):
  if xs.shape != mask.shape:
    return False
  return (torch.exp(xs) * mask) / (torch.exp(xs) * mask).sum(dim=1, keepdim=True)


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C=16, num_classes=10, layers=5, criterion=nn.CrossEntropyLoss().cuda(), steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier


    self.reduce_idx = [layers//3, 2*layers//3]
    self.normal_idx = [i for i in range(layers) if i not in self.reduce_idx]

    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in self.reduce_idx:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = softmax_with_mask(self.alphas_reduce, self.mask_reduce)
      else:
        weights = softmax_with_mask(self.alphas_normal, self.mask_normal)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):

    k = sum(1 for i in range(self._steps) for n in range(2+i))

    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad = True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad = True)

    self.mask_normal = Variable(torch.ones(k, num_ops).cuda(), requires_grad = False)
    self.mask_reduce = Variable(torch.ones(k, num_ops).cuda(), requires_grad = False)

    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

    self.mask = [
      self.mask_normal,
      self.mask_reduce,
    ]

  def set_mask(self, mask):
    for sm, m in zip(self.mask, mask):
      sm.data = m.data
    self.mask_alpha_para()
  
  def get_mask(self):
    return self.mask

  def arch_parameters(self):
    return self._arch_parameters

  def set_alpha(self, alpha):
    for sa, a in zip(self._arch_parameters, alpha):
      sa.data = a.data
  
  # calculate the theoretical size of the network
  def get_model_size(self):
    model_size = 0.
    for name, value in self.state_dict().items():
      if 'cell' not in name or 'preprocess' in name:
        model_size += sys.getsizeof(value.storage())
      else:
        cell_idx = int(re.findall(r'\d+', name)[0])
        [i, j] = list(map(int, re.findall(r'\d+', name)[1: 3]))
        if cell_idx in self.reduce_idx and self.mask_reduce[i][j] == 1.:
          model_size += sys.getsizeof(value.storage())
        elif cell_idx in self.normal_idx and self.mask_normal[i][j] == 1.:
          model_size += sys.getsizeof(value.storage())
    return model_size / 1024 / 1024

  def mask_alpha_para(self):
    self.alphas_normal.data = self.alphas_normal.data * self.mask_normal.data
    self.alphas_reduce.data = self.alphas_reduce.data * self.mask_reduce.data
    for name, value in self.named_parameters():
      if 'cell' not in name or 'preprocess' in name:
        continue
      else:
        cell_idx = int(re.findall(r'\d+', name)[0])
        [i, j] = list(map(int, re.findall(r'\d+', name)[1: 3]))
        if cell_idx in self.reduce_idx and self.mask_reduce[i][j] == 0.:
          value.requires_grad = False
        elif cell_idx in self.normal_idx and self.mask_normal[i][j] == 0.:
          value.requires_grad = False

  def genotype(self):
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(softmax_with_mask(self.alphas_normal, self.mask_normal).data.cpu().numpy())
    gene_reduce = _parse(softmax_with_mask(self.alphas_reduce, self.mask_reduce).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

