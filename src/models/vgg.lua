local dlt = require('dlt._env')

-- Original paper 
-- https://arxiv.org/pdf/1409.1556.pdf
-- Adapted from 
-- https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/vggbn.lua

function dlt.models.vgg(modelType, nClasses,bn,dropout,w,h)
   modelType = modelType or 'A'
   nClasses = nClasses or 205
   bn = bn or true
   dropout = dropout or true
   w = w or 224
   h = h or 224

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 
                                                512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 
                                                    'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 
                                        512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      dlt.log:error('Unknown model type for VGG : ' .. modelType .. 
                        '. Available types: [A,B,D,E]')
   end

   local currentW,currentH = w,h
   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
            currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,
                                                            currentH,2,2,2,2)
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,
                                                                    1,1,1,1);
            currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,
                                                        currentH,3,3,1,1,1,1)
            features:add(conv3)
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end


   local classifier = nn.Sequential()
   classifier:add(nn.View(512*currentW*currentH))
   classifier:add(nn.Linear(512*currentW*currentH, 4096))
   classifier:add(nn.ReLU(true))
   if bn then classifier:add(nn.BatchNormalization(4096, 1e-4)) end
   if dropout then classifier:add(nn.Dropout(0.5)) end
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.ReLU(true))
   if bn then classifier:add(nn.BatchNormalization(4096, 1e-4)) end
   if dropout then classifier:add(nn.Dropout(0.5)) end
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end