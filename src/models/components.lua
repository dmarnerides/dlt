local dlt = require('dlt._env')
dlt.components = {}

-- A bunch of shortcuts for frequently used components 
-- (e.g. Convolution - SpatialBatchNormalization - ReLU - Dropout)
-- At first seemed like a good idea to make building blocks, but it turns out that
-- building models is actually pretty easy, so these shortcuts actually make things
-- slower to implement and not readable.
-- Nevertheless some are useful (e.g. bypass)
local C = dlt.components

local tableCombinations = { add = 'CAddTable', sub = 'CSubTable', mul = 'CMulTable', 
                            div = 'CDivTable', max = 'CMaxTable', min = 'CMinTable' }

-- s = {nInput, nOutput,
--         [net],
--         [full], default is false (i.e. SpatialConvolution and not SpatialFullConvolution)
--         [resize (1,2,3 ...)],[kW], [kH], [dW], [dH], [padW], [padH],[adjW],[adjH]
--         [sbn = table of arguments or nil], 
--         [drop = value or nil], 
--         [act = {name, [arg]} ]}
function C.spatialLayer(s) -- spatial convolution (or fullconvolution), batchnorm and activation
    s.net = s.net or nn.Sequential()
    -- Setup convolutions
    s.kW = s.kW or 3; s.kH = s.kH or s.kW

    if s.full then
        if s.resize then
            s.dW = s.dW or s.resize
            s.dH = s.dH or s.dW
            if (s.dW ~= s.resize) or (s.dH ~= s.resize) then 
                dlt.log:error('For convolutional resizing dW and dH must be equal to resize')
            end
        else s.dW = s.dW or 1; s.dH = s.dH or s.dW end
        s.adjW = (s.kW - s.dW)%2
        s.adjH = (s.kH - s.dH)%2
        if s.resize then 
            s.padW = (s.kW + s.adjW - s.dW)/2
            s.padH = (s.kH + s.adjH - s.dH)/2
        end
        s.net:add(nn.SpatialFullConvolution(s.nInput,s.nOutput,s.kW,s.kH,s.dW,s.dH,s.padW,s.padH,s.adjW,s.adjH))
    else
        if s.resize then
            s.dW = s.dW or s.resize
            s.dH = s.dH or s.dW
            if (s.dW ~= s.resize) or (s.dH ~= s.resize) then 
                dlt.log:error('For convolutional resizing dW and dH must be equal to resize')
            end
        else s.dW = s.dW or 1; s.dH = s.dH or s.dW end
        if s.resize == 1 and (s.kW%2 == 0 or s.kH%2 == 0) then
            dlt.log:warning('Cannot keep same size accurately with even kernel. Setting resize to nil.')
            s.resize = nil
        end
        if not s.padW then if s.resize then s.padW = torch.ceil((s.kW)/2) - 1  else s.padW = 0 end end
        if not s.padH then if s.resize then s.padH = torch.ceil((s.kH)/2) - 1 else s.padH = 0 end end
        s.net:add(nn.SpatialConvolution(s.nInput,s.nOutput,s.kW,s.kH,s.dW,s.dH,s.padW,s.padH))
    end
    if s.sbn then s.net:add(nn.SpatialBatchNormalization(s.nOutput,unpack(s.sbn))) end
    if s.drop then s.net:add(nn.Dropout(s.drop)) end
    if s.act then s.act.arg = s.act.arg or {} end
    if s.act then s.net:add(nn[s.act.name](unpack(s.act.arg))) end
    return s.net
end

-- Shortcuts for useful components
-- if full then using SpatialFullConvolution, dimensions will be upsampled with resize, otherwise they will be downsampled with SpatialConvolution
function C.Conv(nInput,nOutput,kernel,resize,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full}
end
function C.ConvSbn(nInput,nOutput,kernel,resize,sbnArg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg}
end
function C.ConvSbnDrop(nInput,nOutput,kernel,resize,sbnArg,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg,drop = drop}
end
function C.ConvSbnReLU(nInput,nOutput,kernel,resize,sbnArg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'ReLU', arg = {true}}}
end
function C.ConvSbnLReLU(nInput,nOutput,kernel,resize,sbnArg,lreluarg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'LeakyReLU', arg = lreluarg}}
end
function C.ConvSbnTanh(nInput,nOutput,kernel,resize,sbnArg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'Tanh', arg = {true}}}
end
function C.ConvSbnDropReLU(nInput,nOutput,kernel,resize,sbnArg,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'ReLU', arg = {true}},drop = drop}
end
function C.ConvSbnDropLReLU(nInput,nOutput,kernel,resize,sbnArg,drop,lreluarg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'LeakyReLU', arg = lreluarg},drop = drop}
end
function C.ConvSbnDropTanh(nInput,nOutput,kernel,resize,sbnArg,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, sbn = sbnArg, act = {name = 'Tanh', arg = {true}},drop = drop}
end
function C.ConvReLU(nInput,nOutput,kernel,resize,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'ReLU', arg = {true}}}
end
function C.ConvLReLU(nInput,nOutput,kernel,resize,lreluarg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'LeakyReLU', arg = lreluarg}}
end
function C.ConvTanh(nInput,nOutput,kernel,resize,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'Tanh', arg = {true}}}
end
function C.ConvDropReLU(nInput,nOutput,kernel,resize,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'ReLU', arg = {true}},drop = drop}
end
function C.ConvDropLReLU(nInput,nOutput,kernel,resize,drop,lreluarg,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'LeakyReLU', arg = lreluarg},drop = drop}
end
function C.ConvDropTanh(nInput,nOutput,kernel,resize,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full, act = {name = 'Tanh', arg = {true}},drop = drop}
end
function C.ConvDrop(nInput,nOutput,kernel,resize,drop,full,net)
    return dlt.components.spatialLayer{nInput = nInput,nOutput=nOutput,net=net, resize = resize, kW = kernel,full = full,drop = drop}
end

function C.residual(nInput,nMid,convSize)
    convSize = convSize or 3
    nMid = nMid or nInput
    local net =  dlt.components.spatialLayer{ nInput = nInput, nOutput = nMid, resize =  1, kW = convSize, sbn = {1e-4}, act = {name = 'ReLU', arg = {true}} }
    dlt.components.spatialLayer{net = net, nInput = nMid, nOutput = nInput, resize =  1, kW = convSize, sbn = {1e-4} }
    return dlt.components.bypass(net)
end


function C.fuseConvolutionalFeatures(fine,coarse,wRep,hRep)
    local wm = nn.ConcatTable() for i=1,wRep do wm:add(nn.Identity()) end
    local hm = nn.ConcatTable() for i=1,hRep do hm:add(nn.Identity()) end
    coarse:add(wm):add(nn.JoinTable(2,3)):add(hm):add(nn.JoinTable(3,3))
    return nn.Sequential():add(nn.ConcatTable():add(fine):add(coarse)):add(nn.JoinTable(2))
end

function C.multiplyAdd(module)
    local mg = module
    local ag = module:clone()
    local mul = C.bypass(mg,'mul')
    local add = C.bypass(mul,'add',ag)
    return C.bypass(add,'add')
end

function C.bypass(net,tableCombine,shortcut)
    shortcut = shortcut or nn.Identity()
    tableCombine = tableCombine or 'add'
    local cat = nn.ConcatTable():add(net):add(shortcut)
    local ret = nn.Sequential():add(cat):add(nn[tableCombinations[tableCombine]](true))
    return ret
end

function C.fire(inChannels, midChannels, outChannels1, outChannels2)
    local net = nn.Sequential()
                    :add(nn.SpatialConvolution(inChannels, midChannels, 1, 1)) 
                    :add(nn.ReLU(true))
    local exp = nn.Concat(2)
                    :add(nn.SpatialConvolution(midChannels, outChannels1, 1, 1))
                    :add(nn.SpatialConvolution(midChannels, outChannels2, 3, 3, 1, 1, 1, 1))
    return net:add(exp):add(nn.ReLU(true))
end