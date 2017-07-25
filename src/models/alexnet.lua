local dlt = require('dlt._env')

-- Adapted from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/alexnet.lua
-- Note, parallelism was removed, 
-- but if using multiGPU then DPT will wrap the whole model

local function makeFeatures(w,h,inChannels,featureList,bn)
    local currentW,currentH = w,h
    local feat = nn.Sequential()
    -- 224 -> 55
    feat:add(nn.SpatialConvolution(inChannels,featureList[1],11,11,4,4,2,2))
    if bn then feat:add(nn.SpatialBatchNormalization(featureList[1],1e-4)) end
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,
                                                                11,11,4,4,2,2)
    feat:add(nn.ReLU(true))
    -- 55 ->  27
    feat:add(nn.SpatialMaxPooling(3,3,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,
                                                                    3,3,2,2)
    --  27 -> 27
    feat:add(nn.SpatialConvolution(featureList[1],featureList[2],5,5,1,1,2,2))
    if bn then feat:add(nn.SpatialBatchNormalization(featureList[2],1e-4)) end
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,
                                                                5,5,1,1,2,2)
    feat:add(nn.ReLU(true))
    --  27 ->  13
    feat:add(nn.SpatialMaxPooling(3,3,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,
                                                                    3,3,2,2)
    --  13 ->  13
    feat:add(nn.SpatialConvolution(featureList[2],featureList[3],3,3,1,1,1,1))
    if bn then feat:add(nn.SpatialBatchNormalization(featureList[3],1e-4)) end
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,
                                                                3,3,1,1,1,1)
    feat:add(nn.ReLU(true))
    --  13 ->  13
    feat:add(nn.SpatialConvolution(featureList[3],featureList[4],3,3,1,1,1,1))
    if bn then feat:add(nn.SpatialBatchNormalization(featureList[4],1e-4)) end
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,
                                                                3,3,1,1,1,1)
    feat:add(nn.ReLU(true))
    --  13 ->  13
    feat:add(nn.SpatialConvolution(featureList[4],featureList[5],3,3,1,1,1,1))
    if bn then feat:add(nn.SpatialBatchNormalization(featureList[5],1e-4)) end
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH
                                                                ,3,3,1,1,1,1)
    feat:add(nn.ReLU(true))
    -- 13 -> 6
    feat:add(nn.SpatialMaxPooling(3,3,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH
                                                                    ,3,3,2,2)
    return feat, currentW, currentH
end

local function makeClassifier(currentW,currentH,nClasses,dropout,bn)
    local classifier = nn.Sequential()
    classifier:add(nn.View(256*currentW*currentH))
    if dropout then classifier:add(nn.Dropout(0.5)) end
    classifier:add(nn.Linear(256*currentW*currentH, 4096))
    if bn then classifier:add(nn.BatchNormalization(4096, 1e-4)) end
    classifier:add(nn.ReLU(true))
    if dropout then classifier:add(nn.Dropout(0.5)) end
    classifier:add(nn.Linear(4096, 4096))
    if bn then classifier:add(nn.BatchNormalization(4096, 1e-4)) end
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(4096, nClasses))
    classifier:add(nn.LogSoftMax())
    return classifier
end

function dlt.models.alexnet(w,h,inChannels,nClasses)
    w = w or 224
    h = h or 224
    inChannels = inChannels or 3
    nClasses = nClasses or 1000
    local features = nn.Concat(2)
     -- branch 1
    local fb1,currentW,currentH = makeFeatures(w,h,inChannels,
                                                {48,128,192,192,128},false)
    -- branch 2
    local fb2 = fb1:clone() 
    -- reset branch 2's weights
    for k,v in ipairs(fb2:findModules('nn.SpatialConvolution')) do
        v:reset() 
    end 
    features:add(fb1):add(fb2)
    -- 1.3. Create Classifier (fully connected layers)
    local classifier = makeClassifier(currentW,currentH,nClasses,true,false)
    -- 1.4. Combine 1.1 and 1.3 to produce final model
    local model = nn.Sequential():add(features):add(classifier)
    return model
end

-- this is AlexNet that was presented in the One Weird Trick paper.
--  http://arxiv.org/abs/1404.5997
function dlt.models.alexnet2(w,h,inChannels,nClasses)
    w = w or 224
    h = h or 224
    inChannels = inChannels or 3
    nClasses = nClasses or 1000
    local features,currentW,currentH = makeFeatures(w,h,inChannels,
                                                {64,192,384,256,256},true)
    local classifier = makeClassifier(currentW,currentH,nClasses,true,true)
    local model = nn.Sequential():add(features):add(classifier)
    return model
end
