-- https://arxiv.org/pdf/1505.04597.pdf
-- https://arxiv.org/pdf/1611.07004v1.pdf

local dlt = require('dlt._env')

function dlt.models.unet(layers,nInput,nOutput,nDrop,noPadding,noTanh,useSbn)
    if useSbn == nil then useSbn = true end
    local pad = noPadding and 0 or 1
    if not layers then 
        dlt.log:error('Unet requires specification of layers.') 
    end
    nOutput = nOutput or 3
    nInput = nInput or 3
    nDrop = nDrop or 3
    local function LSS(nIn,nOut,sbn)
        if sbn == nil then sbn = true end
        local ret =  nn.Sequential()
        ret:add(nn.LeakyReLU(0.2,true))
           :add(nn.SpatialConvolution(nIn, nOut, 4, 4, 2, 2, pad, pad))
        if sbn then ret:add(nn.SpatialBatchNormalization(nOut)) end
        return ret
    end
    local dropCount = 1
    
    local function RFS(nIn,nOut,sbn)
        if sbn == nil then sbn = true end
        local ret = nn.Sequential()
        ret:add(nn.ReLU(true))
           :add(nn.SpatialFullConvolution(nIn, nOut, 4, 4, 2, 2, pad, pad))
        if sbn then ret:add(nn.SpatialBatchNormalization(nOut)) end
        if dropCount <= nDrop then 
            ret:add(nn.Dropout(0.5)) 
            dropCount = dropCount + 1 
        end
        return ret
    end

    local function recurse(current,next,...)
        if #{...} == 0 then 
            return nn.Sequential()
                :add(nn.ConcatTable()
                        :add( nn.Sequential()
                                :add(LSS(current,next,false))
                                :add(RFS(next,current)) 
                            )
                        :add(nn.Identity()) 
                    )
                :add(nn.JoinTable(1,3))
        else
            return nn.Sequential()
                :add(nn.ConcatTable()
                        :add( nn.Sequential()
                                :add(LSS(current,next))
                                :add(recurse(next,...))
                                :add(RFS(next*2,current)) 
                            )
                        :add(nn.Identity()) 
                    )
                :add(nn.JoinTable(1,3))
        end
    end

    local model = nn.Sequential()
        :add(nn.SpatialConvolution(nInput, layers[1], 4, 4, 2, 2, pad, pad))
        :add(recurse(unpack(layers)))
        :add(nn.ReLU(true))
        :add(nn.SpatialFullConvolution(layers[1]*2, nOutput, 4, 4, 2, 2, 
                                                                  pad, pad))
    if not noTanh then model:add(nn.Tanh(true)) end

    return model
end