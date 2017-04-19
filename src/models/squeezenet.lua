local dlt = require('dlt._env')

-- Adapted from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/squeezenet.lua

-- implementation of squeezenet proposed in: http://arxiv.org/abs/1602.07360
local fire = dlt.components.fire
local bypass = dlt.components.bypass

function dlt.models.squeezenet(nClasses,LogSoftMax)
    nClasses = nClasses or 1000 
    LogSoftMax = LogSoftMax or false
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 0, 0)) -- conv1
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    net:add(fire(96, 16, 64, 64))  --fire2
    net:add(bypass(fire(128, 16, 64, 64)))  --fire3
    net:add(fire(128, 32, 128, 128))  --fire4
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    net:add(bypass(fire(256, 32, 128, 128)))  --fire5
    net:add(fire(256, 48, 192, 192))  --fire6
    net:add(bypass(fire(384, 48, 192, 192)))  --fire7
    net:add(fire(384, 64, 256, 256))  --fire8
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    net:add(bypass(fire(512, 64, 256, 256)))  --fire9
    net:add(nn.Dropout())
    net:add(nn.SpatialConvolution(512, nClasses, 1, 1, 1, 1, 1, 1)) --conv10
    net:add(nn.ReLU(true))
    net:add(nn.SpatialAveragePooling(14, 14, 1, 1))
    net:add(nn.View(nClasses))
    if LogSoftMax then net:add(nn.LogSoftMax()) else net:add(nn.Sigmoid()) end
    return net
end