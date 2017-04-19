local dlt = require('dlt._env')

-- Adapted from the torch 60 minute tutorial
function dlt.models.lenet5(w,h,inChannels,nClasses)
    inChannels = inChannels or 1
    nClasses = nClasses or 10
    w = w or 32
    h = h or 32

    local currentW,currentH = w,h
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(inChannels, 6, 5, 5)) -- inChannels input image channels, 6 output channels, 5x5 convolution kernel
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,5,5)

    net:add(nn.ReLU(true))                       -- non-linearity 
    net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,2,2,2,2)

    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,5,5)

    net:add(nn.ReLU(true))                       -- non-linearity 
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,2,2,2,2)
    
    net:add(nn.View(16*currentW*currentH))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(16*currentW*currentH, 120))             -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.ReLU(true))                       -- non-linearity 
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU(true))                       -- non-linearity 
    net:add(nn.Linear(84, nClasses))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.Sigmoid())
    return net 
end

