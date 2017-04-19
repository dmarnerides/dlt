-- https://arxiv.org/pdf/1505.04597.pdf
-- https://arxiv.org/pdf/1611.07004v1.pdf

local dlt = require('dlt._env')

function dlt.models.unet(layers,nInput,nOutput,nDrop)
    if not layers then dlt.log:error('Unet requires specification of layers (e.g. {32,64,128}) ') end
    nOutput = nOutput or 3
    nInput = nInput or 3
    nDrop = nDrop or 3
    local function LSS(nIn,nOut)
        return nn.Sequential()
                    :add(nn.LeakyReLU(0.2,true))
                    :add(nn.SpatialConvolution(nIn, nOut, 4, 4, 2, 2, 1, 1))
                    :add(nn.SpatialBatchNormalization(nOut))
    end
    local dropCount = 1
    
    local function RFS(nIn,nOut)
        local ret = nn.Sequential()
                    :add(nn.ReLU(true))
                    :add(nn.SpatialFullConvolution(nIn, nOut, 4, 4, 2, 2, 1, 1))
                    :add(nn.SpatialBatchNormalization(nOut))
        if dropCount <= nDrop then ret:add(nn.Dropout(0.5)) dropCount = dropCount + 1 end
        return ret
    end

    local function recurse(current,next,...)
        if #{...} == 0 then 
            return nn.Sequential():add(nn.ConcatTable():add( nn.Sequential():add(LSS(current,next)):add(RFS(next,current))  )
                                                        :add(nn.Identity()) ):add(nn.JoinTable(1,3))
        else
            return nn.Sequential():add(nn.ConcatTable():add( nn.Sequential():add(LSS(current,next)):add(recurse(next,...)):add(RFS(next*2,current)) )
                                    :add(nn.Identity()) ):add(nn.JoinTable(1,3))
        end
    end

    local model = nn.Sequential()
                    :add(nn.SpatialConvolution(nInput, layers[1], 4, 4, 2, 2, 1, 1))
                    :add(recurse(unpack(layers)))
                    :add(nn.ReLU(true))
                    :add(nn.SpatialFullConvolution(layers[1]*2, nOutput, 4, 4, 2, 2, 1, 1))
                    :add(nn.ReLU(true))
    dlt.components.ConvTanh(nInput,nInput,3,1,false,model)

    return model
end