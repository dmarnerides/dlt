local dlt = require('dlt._env')
require('torch')
require('paths')
require('class')
require('optim')
require('nn')
threads = require('threads')
dlt.models = {}
dlt.have = {}
for i,mod in pairs({'cutorch','cunn','cudnn','image','csvigo'}) do dlt.have[mod] =  pcall(require,mod) end

local modules = {
    'logger',
    'helper',
    'slurm',
    'color',
    'settings',
    'components',
    'colornet', 'lenet5', 'alexnet',
    'squeezenet', 'unet', 'vgg',
    'model',
    'donkey',
    'data',
    'loader',
    'pix2pix',
    'places',
    'celeba',
    'cifar',
    'mnist',
    'optimizer',
    'trainer',
    'trainlog',
    'dispatcher'
}

for i,mod in ipairs(modules) do require('dlt.' .. mod) end
-- Make dlt log
dlt.log = dlt.Logger()
return dlt