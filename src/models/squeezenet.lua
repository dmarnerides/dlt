local dlt = require('dlt._env')

-- Squeezenet paper: http://arxiv.org/abs/1602.07360

-- Adapted from 
-- https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/squeezenet.lua

local function bypass(net)
    local cat = nn.ConcatTable():add(net):add(nn.Identity())
    local ret = nn.Sequential():add(cat):add(nn.CAddTable(true))
    return ret
end

local function fire(inChannels, midChannels, outChannels1, outChannels2)
    local net = nn.Sequential()
                    :add(nn.SpatialConvolution(inChannels, midChannels, 
                                                                1, 1)) 
                    :add(nn.ReLU(true))
    local exp = nn.Concat(2)
                    :add(nn.SpatialConvolution(midChannels, outChannels1, 
                                                                 1, 1))
                    :add(nn.SpatialConvolution(midChannels, outChannels2, 
                                                        3, 3, 1, 1, 1, 1))
    return net:add(exp):add(nn.ReLU(true))
end

function dlt.models.squeezenet(nClasses,LogSoftMax)
    nClasses = nClasses or 1000
    local net = nn.Sequential()
    -- conv1
    net:add(nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 0, 0)) 
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    --fire2
    net:add(fire(96, 16, 64, 64))
    --fire3
    net:add(bypass(fire(128, 16, 64, 64)))  
    --fire4
    net:add(fire(128, 32, 128, 128))  
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    --fire5
    net:add(bypass(fire(256, 32, 128, 128)))  
    --fire6
    net:add(fire(256, 48, 192, 192))  
    --fire7
    net:add(bypass(fire(384, 48, 192, 192)))  
    --fire8
    net:add(fire(384, 64, 256, 256))  
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    --fire9
    net:add(bypass(fire(512, 64, 256, 256)))  
    net:add(nn.Dropout())
    --conv10
    net:add(nn.SpatialConvolution(512, nClasses, 1, 1, 1, 1, 1, 1)) 
    net:add(nn.ReLU(true))
    net:add(nn.SpatialAveragePooling(14, 14, 1, 1))
    net:add(nn.View(nClasses))
    if LogSoftMax then 
        net:add(nn.LogSoftMax()) 
    else 
        net:add(nn.Sigmoid()) 
    end
    return net
end