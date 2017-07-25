local dlt = require('dlt._env')

-- Code adapted from https://github.com/soumith/dcgan.torch

local function getClosures(net,sbn)
    local function Conv(nIn,nOut)
        net:add(nn.SpatialConvolution(nIn,nOut,4,4,2,2,1,1))
    end
    local function FConv(nIn,nOut)
        net:add(nn.SpatialFullConvolution(nIn,nOut,4,4,2,2,1,1))
    end
    local function SBN(nf)
        if sbn then net:add(nn.SpatialBatchNormalization(nf)) end
    end
    local function LReLU() net:add(nn.LeakyReLU(0.2, true)) end
    local function ReLU() net:add(nn.ReLU(true)) end
    return Conv,SBN,LReLU,FConv,ReLU
end

dlt.models.dcgan = {}

function dlt.models.dcgan.initWeights(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m:noBias()
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end

function dlt.models.dcgan.generator64(nz,ngf,nc,init,sbn)
    nz = nz or 100
    ngf = ngf or 64
    nc = nc or 3
    if init == nil then init = true end
    local netG = nn.Sequential()
    _,SBN,_,FConv,ReLU = getClosures(netG,sbn)

    netG:add(nn.View(nz,1,1))
    
    netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
    SBN(ngf*8)
    ReLU()
    -- state size: (ngf*8) x 4 x 4
    FConv(ngf * 8, ngf * 4)
    SBN(ngf*4)
    ReLU()
    -- state size: (ngf*4) x 8 x 8
    FConv(ngf * 4, ngf * 2)
    SBN(ngf*2)
    ReLU()
    -- state size: (ngf*2) x 16 x 16
    FConv(ngf * 2, ngf)
    SBN(ngf)
    ReLU()
    -- state size: (ngf) x 32 x 32
    FConv(ngf, nc)
    netG:add(nn.Tanh()) 
    -- state size: (nc) x 64 x 64
    if init then netG:apply(dlt.models.dcgan.initWeights) end
    return netG
end

function dlt.models.dcgan.generator32(nz,ngf,nc,init,sbn)
    nz = nz or 100
    ngf = ngf or 64
    nc = nc or 3
    if init == nil then init = true end
    local netG = nn.Sequential()
    _,SBN,_,FConv,ReLU = getClosures(netG,sbn)

    netG:add(nn.View(nz,1,1))
    
    netG:add(nn.SpatialFullConvolution(nz, ngf * 4, 4, 4))
    SBN(ngf*4)
    ReLU()
        -- state size: (ngf*8) x 4 x 4
    FConv(ngf * 4, ngf * 2)
    SBN(ngf*2)
    ReLU()
        -- state size: (ngf*4) x 8 x 8
    FConv(ngf * 2, ngf)
    SBN(ngf)
    ReLU()
        -- state size: (ngf*2) x 16 x 16
    FConv(ngf, nc)
    netG:add(nn.Tanh()) 
        -- state size: (nc) x 32 x 32
    if init then netG:apply(dlt.models.dcgan.initWeights) end
    return netG
end

function dlt.models.dcgan.discriminator64(ndf,nc,sigmoid,init,sbn)
    ndf = ndf or 64
    nc = nc or 3
    if init == nil then init = true end
    local netD = nn.Sequential()
    Conv,SBN,LReLU,_ = getClosures(netD,sbn)
    -- input is (nc) x 64 x 64
    Conv(nc, ndf)
    LReLU()
    -- state size: (ndf) x 32 x 32
    Conv(ndf, ndf * 2)
    SBN(ndf*2)
    LReLU()
    -- state size: (ndf*2) x 16 x 16
    Conv(ndf * 2, ndf * 4)
    SBN(ndf*4)
    LReLU()
    -- state size: (ndf*4) x 8 x 8
    Conv(ndf * 4, ndf * 8)
    SBN(ndf*8)
    LReLU()
    -- state size: (ndf*4) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
    if sigmoid then netD:add(nn.Sigmoid()) end
        -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
        -- state size: 1
    if init then netD:apply(dlt.models.dcgan.initWeights) end
    return netD
end

function dlt.models.dcgan.discriminator32(ndf,nc,sigmoid,init,sbn)
    ndf = ndf or 64
    nc = nc or 3
    if init == nil then init = true end
    local netD = nn.Sequential()
    Conv,SBN,LReLU,_ = getClosures(netD,sbn)
    -- input is (nc) x 32 x 32
    Conv(nc, ndf)
    LReLU()
    -- state size: (ndf*2) x 16 x 16
    Conv(ndf, ndf * 2)
    SBN(ndf*2)
    LReLU()
    -- state size: (ndf*4) x 8 x 8
    Conv(ndf * 2, ndf * 4)
    SBN(ndf*4)
    LReLU()
    -- state size: (ndf*4) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 4, 1, 4, 4))
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3)) 
    if sigmoid then netD:add(nn.Sigmoid()) end
        -- state size: 1
    if init then netD:apply(dlt.models.dcgan.initWeights) end
    return netD
end