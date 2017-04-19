local dlt = require('dlt._env')

-- Helper functions!
-- Many were written for functionalities that were removed/changed but left them here just in case
-- Almost surely not bug free
dlt.help = {}
local H = dlt.help

function H.writeTableToFile(fName,t)
    local file = io.open(fName,'w+')
    for key,val in pairs(t) do
        file:write(string.format('%s\t\t%s\n',tostring(key),tostring(val)))
    end
    file:close()
end

-- Removes . and .. directories from array
function H.clean(files)
    for i = #files,1,-1 do
        if files[i]:sub(-1,-1) == '.' then 
            table.remove(files,i) 
        end
    end
    return files
end

function H.fullPath(directory,files)
    local ret = {}
    for i,val in pairs(files) do
        ret[i] = paths.concat(directory,files[i])
    end
    return ret
end

function H.mergeArrays(l1,l2)
    if #l1 == 0 then return l2 end
    if #l2 == 0 then return l1 end
    for _,val in ipairs(l2) do
        table.insert(l1,val)
    end    
    return l1
end

function H.getFiles(directory,extensions,recursive)
    local fileList = H.fullPath(directory,H.clean(paths.dir( directory )))
    local ret = H.clean(paths.dir( directory ))
    for i = #ret,1,-1 do
       if paths.dirp(ret[i]) or not extensions[ret[i]:sub(-3,-1)] then table.remove(ret,i)end
    end
    ret = H.fullPath(directory,ret)
    if recursive then
        for i = 1,#fileList do
                ret = paths.dirp(fileList[i]) and H.mergeArrays(ret,H.getFiles(fileList[i],extensions,recursive),fileList[i]) or ret
        end
    end
    return ret
end

function H.readList(fileName)
    local file = io.open(fileName, "r");
    local ret = {}
    for line in file:lines() do
        table.insert (ret, line);
    end
    return ret
end

function H.execute(command)
    local f = assert(io.popen(command)) -- runs command
    local l = f:read("*a") -- read output of command
    f:close()
    local ret = l:gsub("\n$", "")
    return ret;
end

function H.split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then return { str } end

    local result,pat,lastpos = {},"(.-)" .. delim .. "()",nil
    for part, pos in string.gfind(str, pat) do table.insert(result, part); lastPos = pos; end
    table.insert(result, string.sub(str, lastPos))
    return result
end

function H.checkMakeDir(path) 
    if not paths.dirp(path) and not paths.mkdir(path) then
        dlt.log:error('Unable to create directory: ' .. path .. '\n')
    end
end

function H.checkHomePath(path)
    path = path:match('^~/?') and paths.concat(paths.home,path:sub(3,-1)) or path
    return path
end

function H.boolify(val) return val == 'true' end

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
        half = 'createCudaHostHalfTensor',
        float = 'createCudaHostTensor',
        double = 'createCudaHostDoubleTensor'
    }
}

function H.inTable(t,v)
    for _,x in pairs(t) do if x == v then return true end end
    return false
end

function H.createBatch(batchSize, sizes,tensorType, nGPU, pinned)
    if not sizes then dlt.log:error('dataPoint sizes of experiment not provided for creation of batch.') end
    tensorType = tensorType or 'float'
    nGPU = nGPU or 0
    local device = nGPU == 0 and 'cpu' or 'gpu'    
    local supportPinned = device == 'gpu'
    if H.inTable({'byte','char','short','int','long'},tensorType) then supportPinned = false end

    local retType = {}
    -- Pinned
    -- for name,conf in pairs(sizes) do
    --     local makePinned = pinned and supportPinned
    --     retType[name] = makePinned and cutorch[H.tensorList[pinned][tensorType]] or torch[H.tensorList[device][tensorType]]
    -- end
    
    for name,conf in pairs(sizes) do retType[name] = torch[H.tensorList[device][tensorType]] end

    -- Create batch from sizes
    local ret = {}
    for name,conf in pairs(sizes) do
        if #conf == 0 then -- Classifier data described by empty table
            ret[name] = retType[name](batchSize)
        elseif torch.type(conf[1]) == 'table' then -- Subtables
            ret[name] = {}
            for i, val in ipairs(conf) do
                ret[name][i] = retType[name](batchSize,unpack(val))
            end
        else
            ret[name] = retType[name](batchSize,unpack(conf))
        end
    end
    return ret 
end

function H.logGPUMemory(devID,loggerF)
    local freeMemory, totalMemory = cutorch.getMemoryUsage(devID)
    local div = 1024*1024*1024
    dlt.log[loggerF](dlt.log,string.format('GPU %d: total - %.3fGB, free - %.3fGB.',
                devID,totalMemory/div, freeMemory/div))
end

function H.logAllGPUMemory(nGPU,loggerF)
    for i = 1, nGPU do H.logGPUMemory(i,loggerF) end
end

function H.copyPoint(fromPoint,toPoint)
    for name,data in pairs(fromPoint) do
        if torch.type(data) == 'table' then
            for subname,subdata in pairs(data) do
                if toPoint[name][subname].copy then toPoint[name][subname]:copy(subdata)
                else toPoint[name][subname] = subdata end
            end
        else
            if toPoint[name].copy then toPoint[name]:copy(data)
            else toPoint[name] = data end
        end
    end
end

function H.hasNaN(t) return t:ne(t):sum() ~= 0 end

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
-- Assumptions 0 < a < b
function H.normalizeInPlace(t,a,b)
    a = a or 0
    b = b or 1
    local tmin,tmax = t:min(),t:max()
    return t:add(-tmin):div(math.max((tmax-tmin)/(b-a),1e-4)):add(a)
end
function H.normalize(t,a,b) 
    return H.normalizeInPlace(t:clone(),a,b) 
end

-- Assumptions: 0 <= clampA < clampB <=1, see normalizeInPlace
function H.clampAndNormalizeInPlace(t,clampA,clampB,a,b) return H.normalizeInPlace(t:clamp(clampA,clampB),a,b) end 
function H.clampAndNormalize(t,clampA,clampB,a,b) return H.clampAndNormalizeInPlace(t:clone(),clampA,clampB,a,b) end

-- Mean squared difference
function H.mseInPlace(t1,t2) return torch.sum(t1:add(-t2):pow(2):div(torch.numel(t1))) end
function H.mse(t1,t2) return H.mseInPlace(t1:clone(),t2) end

-- PSNR
function H.psnrInPlace(t1,t2) return -(10/torch.log(10))*torch.log(H.mseInPlace(t1,t2)) end
function H.psnr(t1,t2) return -(10/torch.log(10))*torch.log(H.mse(t1,t2)) end

-- Assumptions, has 3 dimensions, w,h are less than t's w and h
-- returns a copy
function H.randomCrop(t,w,h)
    local wstart, hstart = torch.random(t:size(2) - w + 1), torch.random(t:size(3) - h + 1)
    return t[{{},{wstart, wstart + w - 1},{hstart,hstart + h - 1}}]:clone()
end
-- Also returns a copy
function H.hflipAndRandomCrop(t,w,h,p)
    p = p or 0.5
    local ret = torch.uniform() < p and image.hflip(t) or t
    return H.randomCrop(ret,w,h)
end

function H.randomHFlip(t,p)
    p = p or 0.5
    local ret = torch.uniform() < p and image.hflip(t) or t
    return ret
end

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

return H