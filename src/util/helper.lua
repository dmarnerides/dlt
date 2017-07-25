local dlt = require('dlt._env')

-- Helper functions
dlt.help = {}
local H = dlt.help

-- Checks if path exists, tries to create it if not
-- Returns useful error and terminates if it can not create it
function H.checkMakeDir(path) 
    if not paths.dirp(path) and not paths.mkdir(path) then
        dlt.log:error('Unable to create directory: ' .. path .. '\n')
    end
end

-- Checks if path given is or starts with '~'
-- and replaces '~' with full path
function H.checkHomePath(path)
    path = path:match('^~/?') and paths.concat(paths.home,path:sub(3,-1)) 
                               or path
    return path
end

-- Handling of types
H.tensorList = {
    cpu = {
        byte = 'ByteTensor',
        char = 'CharTensor',
        short = 'ShortTensor',
        int = 'IntTensor',
        long = 'LongTensor',
        half = 'HalfTensor',
        float = 'FloatTensor',
        double = 'DoubleTensor'
    },
    gpu = {
        byte = 'CudaByteTensor',
        char = 'CudaCharTensor',
        short = 'CudaShortTensor',
        int = 'CudaIntTensor',
        long = 'CudaLongTensor',
        half = 'CudaHalfTensor',
        float = 'CudaTensor',
        double = 'CudaDoubleTensor'
    },
    pinned = {
        byte = 'createCudaHostByteTensor',
        int = 'createCudaHostIntTensor',
        long = 'createCudaHostLongTensor',
        half = 'createCudaHostHalfTensor',
        float = 'createCudaHostTensor',
        double = 'createCudaHostDoubleTensor'
    }
}

local typeList = {'byte','char','short','int',
                  'long','half','float','double'}
local pinnedList = {'byte','int', 'long','half','float','double'}

-- Checks if value v is in table t
function H.inTable(t,v)
    for _,x in pairs(t) do 
        if x == v then 
            return true
        end 
    end
    return false
end

-- Apply a function to all elements in array
function H.apply(t,f)
    for i,v in ipairs(t) do 
        t[i] = f(v) 
    end 
    return t 
end

-- Creates a batch of size batchSize with given dimensions
-- dimensions must be a table of names with corresponding dimensions
--      e.g. dimensions = {input = {3,32,32}, output = {}}
--      If the corresponding dimensions is an empty table ({}) e.g. above,
--      the output member will just be one dimensional of size batchSize.
--      This is used for compatibility with classification criteria
--      dimensions can be tables (up to one level deep)
--      e.g. for models which take a table input we might have
--      dimensions = { input = { {3,32,32}, {3,64,64} } }
-- tensorType can be of byte,char,short,int,long,half,float,double
--      defaults to float
-- device can be 'cpu' or 'gpu'
function H.createBatch(batchSize, dimensions,tensorType, device, pinned)
    -- Check dimensions
    if not dimensions then 
        dlt.log:error('dataPoint dimensions of experiment' .. 
                        ' not provided for creation of batch.') 
    end
    if torch.type(dimensions) ~= 'table' then
        dlt.log:error('dimensions must be a table for createBatch.')
    end
    if next(dimensions) == nil then
        dlt.log:error('dimensions must be a non-empty table for createBatch.')
    end
    -- Default tensor is 'float'
    tensorType = tensorType or 'float'
    if not H.inTable(typeList,tensorType) then
        dlt.log:error('Unsupported type ' .. tensorType) 
    end
    -- Configure device
    device = device or 'cpu'
    if device ~= 'gpu' and device ~= 'cpu' then
        dlt.log:error('Device must be cpu or gpu for createBatch')
    end   
    -- Pinned only if gpu
    local supportPinned = device == 'gpu'
    -- Pinned not supported for short and char
    if pinned and not H.inTable(pinnedList,tensorType) then
        dlt.log:warning('Pinned memory not supported for ' .. tensorType .. 
                           '. Setting to false.' )
        supportPinned = false 
    end

    local retType
    -- Pinned
    if pinned and supportPinned then 
        retType = cutorch[H.tensorList.pinned[tensorType]]
    else
        retType = torch[H.tensorList[device][tensorType]]
    end
    -- Create batch from dimensions
    local ret = {}
    for name,conf in pairs(dimensions) do
        if #conf == 0 then -- Classifier data described by empty table
            ret[name] = retType(batchSize)
        elseif torch.type(conf[1]) == 'table' then -- Subtables
            ret[name] = {}
            for i, val in ipairs(conf) do
                ret[name][i] = retType(batchSize,unpack(val))
            end
        else
            ret[name] = retType(batchSize,unpack(conf))
        end
    end
    return ret 
end

-- Yells gpu memory for devID in GB
function H.logGPUMemory(devID)
    local freeMemory, totalMemory = cutorch.getMemoryUsage(devID)
    local div = 1024*1024*1024
    local str = 'GPU %d: total - %.3fGB, free - %.3fGB.'
    dlt.log:yell(string.format(str, devID,totalMemory/div, freeMemory/div))
end

function H.logAllGPUMemory(nGPU)
    for i = 1, nGPU do H.logGPUMemory(i) end
end

-- Copies a point to another (batches that were created by createBatch)
-- Useful for transfering a batch from cpu to gpu
function H.copyPoint(fromPoint,toPoint)
    for name,data in pairs(fromPoint) do
        if torch.type(data) == 'table' then
            for subname,subdata in pairs(data) do
                if toPoint[name][subname].copy then 
                    -- toPoint[name][subname]:resize(subdata:size()):copy(subdata)
                    toPoint[name][subname]:copy(subdata)
                else 
                    toPoint[name][subname] = subdata 
                end
            end
        else
            if toPoint[name].copy then 
                -- toPoint[name]:resize(data:size()):copy(data)
                toPoint[name]:copy(data)
            else 
                toPoint[name] = data 
            end
        end
    end
end

-- Returns true if tensor has NaN
function H.hasNaN(t) return t:ne(t):sum() ~= 0 end

-- Returns the resulting dimensions of a spatial convolution
function H.SpatialConvolutionSize(width,height,kW,kH,dW,dH,padW,padH)
    dW = dW or 1
    dH = dH or 1
    padW = padW or 0
    padH = padH or 0
    local owidth  = torch.floor((width  + 2*padW - kW) / dW + 1)
    local oheight = torch.floor((height + 2*padH - kH) / dH + 1)
    return owidth,oheight
end
function H.SpatialMaxPoolingSize(width,height,kW,kH,dW,dH,padW,padH)
    return H.SpatialConvolutionSize(width,height,kW,kH,dW,dH,padW,padH)
end

---- tensor transformations
---- If function ends in _ then it is in-place

-- Assumption 0 < a < b
function H.normalize_(t,a,b)
    a = a or 0
    b = b or 1
    local tmin,tmax = t:min(),t:max()
    return t:add(-tmin):div(math.max((tmax-tmin)/(b-a),1e-4)):add(a)
end
function H.normalize(t,a,b) 
    return H.normalize_(t:clone(),a,b) 
end

-- Assumptions: 0 <= clampA < clampB <=1, see normalize_
function H.clampAndNormalize_(t,clampA,clampB,a,b) 
    return H.normalize_(t:clamp(clampA,clampB),a,b) 
end 
function H.clampAndNormalize(t,clampA,clampB,a,b) 
    return H.clampAndNormalize_(t:clone(),clampA,clampB,a,b) 
end

-- Mean squared difference (t1 is changed for the in-place version)
function H.mse_(t1,t2) 
    return torch.sum(t1:add(-t2):pow(2):div(torch.numel(t1))) 
end
function H.mse(t1,t2) 
    return H.mse_(t1:clone(),t2) 
end

-- PSNR (t1 is changed for the in-place version)
function H.psnr_(t1,t2) 
    return -(10/torch.log(10))*torch.log(H.mse_(t1,t2)) 
end
function H.psnr(t1,t2) 
    return -(10/torch.log(10))*torch.log(H.mse(t1,t2)) 
end

-- Assumptions, has 3 dimensions, w,h are less than t's w and h
-- returns a copy
function H.randomCrop(t,w,h)
    local wstart = torch.random(t:size(2) - w + 1) 
    local hstart = torch.random(t:size(3) - h + 1)
    return t[{{},{wstart, wstart + w - 1},{hstart,hstart + h - 1}}]:clone()
end

-- Flips horizontally with probability p and crops randomly
-- returns a copy
function H.hflipAndRandomCrop(t,w,h,p)
    p = p or 0.5
    local ret = torch.uniform() < p and image.hflip(t) or t
    return H.randomCrop(ret,w,h)
end

-- Flips horizontally with probability p
function H.randomHFlip(t,p)
    p = p or 0.5
    local ret = torch.uniform() < p and image.hflip(t) or t
    return ret
end

-- Returns the values that are at low*100% and high*100%
function H.getPercentClamping(img,low,high)
    local imgSize = img:size()
    local npix = imgSize[1]*imgSize[2]*imgSize[3]
    local oned = img:view(npix)
    local lowIndex = low*npix
    if lowIndex < 1 then lowIndex = 1 end
    local highIndex = high*npix
    if highIndex > npix then highIndex = npix end
    local lowRet = oned:kthvalue(lowIndex)
    local highRet = oned:kthvalue(highIndex)
    return lowRet[1], highRet[1]
end



-- local helper functions for getFiles
-- clean removes '.' and '..' from an array of strings
local function clean(files)
    for i = #files,1,-1 do
        if files[i]:sub(-1,-1) == '.' then 
            table.remove(files,i) 
        end
    end
    return files
end
-- calls paths.concat on all files to append directory and
local function fullPath(directory,files)
    local ret = {}
    for i,val in pairs(files) do
        ret[i] = paths.concat(directory,files[i])
    end
    return ret
end
-- Appends array l2 to l1
local function mergeArrays(l1,l2)
    if #l1 == 0 then return l2 end
    if #l2 == 0 then return l1 end
    for _,val in ipairs(l2) do
        table.insert(l1,val)
    end    
    return l1
end

-- getFiles gets all files from directory with given extensions
-- recursively too if flag set
function H.getFiles(directory,extensions,recursive)
    local fileList = fullPath(directory,clean(paths.dir( directory )))
    local ret = clean(paths.dir( directory ))
    for i = #ret,1,-1 do
        if paths.dirp(ret[i]) or not extensions[ret[i]:sub(-3,-1)] then 
            table.remove(ret,i)
        end
    end
    ret = fullPath(directory,ret)
    if recursive then
        for i = 1,#fileList do
            ret = paths.dirp(fileList[i]) and mergeArrays(ret,H.getFiles(fileList[i],extensions,recursive),fileList[i]) or ret
        end
    end
    return ret
end



return H