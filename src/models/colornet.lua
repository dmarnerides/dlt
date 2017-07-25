local dlt = require('dlt._env')


-- Paper: Let there be Color!: 
--        Joint End-to-end Learning of Global and Local Image Priors  
--        for Automatic Image Colorization with Simultaneous Classification
-- http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf

-- Accepts a table with two inputs. 
-- The second is wxh, the first is not constrained 
--    (dimensions should be divisible by 8)
-- Output is a table. 
-- First output is an image of the size of the first input. 
-- Second output is the class probability vector 
-- Trained on 224x224 input 
-- (1 channel in (L from LUV) 
--  2 channels out (ab from Lab - normalized to [0,1]))
function dlt.models.colornet(w,h,nClasses,inChannels,outChannels,useBatchNorm)
    -- Short names and defaults
    local SBatchNorm = nn.SpatialBatchNormalization
    local BatchNorm = nn.BatchNormalization
    w = w or 224
    h = h or 224
    nClasses = nClasses or 205
    inChannels = inChannels or 1
    outChannels = outChannels or 2

    -- Helper function for adding layers
    local function layer(net,inputDepth,outputDepth,stride)
        net:add(nn.SpatialConvolution(inputDepth,outputDepth,
                                        3,3,stride,stride,1,1))
        if useBatchNorm then net:add(SBatchNorm(outputDepth)) end
        net:add(nn.ReLU(true))
    end

    -- Local Features
    local lF = nn.Sequential() 
    layer(lF,inChannels,64,2)
    layer(lF,64,128,1)
    layer(lF,128,128,2)
    layer(lF,128,256,1) 
    layer(lF,256,256,2)
    layer(lF,256,512,1)

    -- Global Features start (before it splits for fusion and classifier)
    local gFstart = nn.Sequential() 
    layer(gFstart,512,512,2)
    layer(gFstart,512,512,1)
    layer(gFstart,512,512,2)
    layer(gFstart,512,512,1)
    -- Change the view for fully connected layers
    local fCLength = w*h/2
    gFstart:add(nn.View(fCLength))
           :add(nn.Linear(fCLength,1024))
    if useBatchNorm then 
        gFstart:add(BatchNorm(1024)) 
    end
    gFstart:add(nn.ReLU(true))
    gFstart:add(nn.Linear(1024,512))
    if useBatchNorm then 
        gFstart:add(BatchNorm(512)) 
    end
    gFstart:add(nn.ReLU(true))

    -- Global Features end for fusion
    local gFend = nn.Sequential()
    gFend:add(nn.Linear(512,256))
    if useBatchNorm then 
        gFend:add(BatchNorm(256)) 
    end
    gFend:add(nn.ReLU(true))
    gFend:add(nn.Replicate(w/8,3))
         :add(nn.Replicate(h/8,4))

    -- Mid Features
    local mF = nn.Sequential() 
    layer(mF,512,512,1)
    layer(mF,512,256,1)

    -- largeNet is the beginning of the convolutional network
    -- where image size is not restricted
    -- up to fusion part
    local largeNet = nn.Sequential()
    largeNet:add(lF)
            :add(mF)

    -- Small is the classifier network (fusion branch)
    local smallNet = nn.Sequential()
    smallNet:add(lF):add(gFstart):add(gFend)

    -- Color model up to fusion part
    local colorModel = nn.Sequential()
    colorModel:add(nn.ParallelTable()
                        :add(largeNet)
                        :add(smallNet))
                :add(nn.JoinTable(2))
    
    -- Complete color model
    layer(colorModel,512,256,1)
    layer(colorModel,256,128,1) 
    colorModel:add(nn.SpatialUpSamplingNearest(2))
    layer(colorModel,128,64,1)
    layer(colorModel,64,64,1)
    colorModel:add(nn.SpatialUpSamplingNearest(2))
    layer(colorModel,64,32,1)
    colorModel:add(nn.SpatialConvolution(32,outChannels,3,3,1,1,1,1))
    colorModel:add(nn.Sigmoid())

    -- Complete classifier
    local classifier = nn.Sequential():add(nn.SelectTable(2))
    classifier:add(lF)
              :add(gFstart)
              :add(nn.Linear(512,256))
    if useBatchNorm then 
        classifier:add(BatchNorm(256)) 
    end
    classifier:add(nn.ReLU(true))
              :add(nn.Linear(256,nClasses))
    if useBatchNorm then 
        classifier:add(BatchNorm(nClasses)) 
    end
    classifier:add(nn.Sigmoid())

    -- Complete model
    local model = nn.ConcatTable()
    model:add(colorModel)
         :add(classifier)

    return model
end