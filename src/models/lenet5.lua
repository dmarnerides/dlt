local dlt = require('dlt._env')

-- Adapted from the torch 60 minute blitz
-- https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb

function dlt.models.lenet5(w,h,inChannels,nClasses)
    inChannels = inChannels or 1
    nClasses = nClasses or 10
    w = w or 32
    h = h or 32

    local currentW,currentH = w,h
    local net = nn.Sequential()
    -- inChannels input image channels,
    -- 6 output channels, 5x5 convolution kernel
    net:add(nn.SpatialConvolution(inChannels, 6, 5, 5))
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,5,5)
    -- non-linearity
    net:add(nn.ReLU(true))
    -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,
                                                                    2,2,2,2)

    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    currentW,currentH = dlt.help.SpatialConvolutionSize(currentW,currentH,5,5)

    -- non-linearity 
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    currentW,currentH = dlt.help.SpatialMaxPoolingSize(currentW,currentH,
                                                                    2,2,2,2)
    
    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.View(16*currentW*currentH))
    -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.Linear(16*currentW*currentH, 120))
    -- non-linearity
    net:add(nn.ReLU(true))
    net:add(nn.Linear(120, 84))
    -- non-linearity
    net:add(nn.ReLU(true))
    -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.Linear(84, nClasses))
    net:add(nn.Sigmoid())
    return net 
end

