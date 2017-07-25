local dlt = require('dlt._env')

local M,parent = torch.class('dlt.Model',dlt)


function M:__init(modelCreate,name,save)
    self.name = name or 'model'
    self.shouldSave = (save == nil) and true or save
    if torch.type(self.name) ~= 'string' then 
        dlt.log:error('Model name must be a string.') 
    end
    self.name = self.name:gsub("^%l", string.upper) 
    dlt.log:section(self.name .. ' initialization')    
    dlt.parse(self)
    dlt.configure(self)
    if torch.type(modelCreate) == 'string' then
        modelCreate = dlt.help.checkHomePath(modelCreate)
        if not paths.filep(modelCreate) then 
            dlt.log:error('Could not find model ' .. modelCreate) 
        end
        self.model = torch.load(modelCreate)
        dlt.log:yell('Loaded ' .. self.name ..   ' from file.');
    elseif torch.type(modelCreate) == 'function' then
        self.model = modelCreate()
        dlt.log:yell('Created ' .. self.name ..   '.') 
    else 
        dlt.log:error('dlt.Model parameter (modelCreate)' .. 
                        ' must be a string or a function.') 
    end

    self:processModel()

    if self.parameters:size():size() ~= 0 then
        dlt.log:yell(string.format('%s parameters: %d.',
                                        self.name, self.parameters:size(1)))
    end
    collectgarbage()
    dlt.log:endSection()
end

function M:getCleanModel()
    -- Clear State
    self.model:clearState()
    -- Remove DataParallelTable
    self.model = torch.type(self.model) == 'nn.DataParallelTable' 
                        and self.model:get(1) 
                         or self.model
    -- Remove cudnn
    if self.useCudnn then 
        cudnn.convert(self.model,nn) 
        cutorch.synchronizeAll()
    end
    -- Return CPU model
    return self:cpu()
end

function M:cpu()
    return self.model:type('torch.' .. dlt.help.tensorList.cpu[self.tensorType])
end
function M:gpu()
    return self.model:type('torch.' .. dlt.help.tensorList.gpu[self.tensorType])
end

function M:processModel()
    -- First cast to cpu
    self:cpu()

    if self.nGPU > 0 then 
        -- Convert to cudnn
        if self.useCudnn then
            cudnn.fastest = self.cudnnFastest
            cudnn.benchmark = self.cudnnBenchmark
            cudnn.verbose = self.cudnnVerbose
            cudnn.convert(self.model,cudnn)
            self:sync()
        end

        -- Transfer to gpu
        self:gpu()
        
        -- If multiple GPUs wrap in DataParallelTable
        if self.nGPU > 1 then            
            local dpt = nn.DataParallelTable(1, self.dptFlatten,self.dptNccl)
                        :add(self.model, torch.range(1, self.nGPU):totable())
            self.model = dpt
            self:sync()
        end
    end
    -- Default to training mode
    self:training()
    -- Reflatten parameters
    self.parameters, self.gradParameters = self.model:getParameters()
end

function M:sync() 
    if self.nGPU > 0 then 
        cutorch.synchronizeAll() 
    end 
end

function M:save(filename)
    if self.shouldSave then 
        torch.save(filename, self:getCleanModel()) 
        self:processModel()
    end
end

function M:training() 
    self.model:training() 
end
function M:evaluate() 
    self.model:evaluate() 
end
function M:zeroGradParameters() 
    self.model:zeroGradParameters() 
end
function M:forward(input) 
    return self.model:forward(input) 
end
function M:backward(input,gradOutput) 
    return self.model:backward(input,gradOutput) 
end
function M:updateGradInput(input,gradOutput) 
    return self.model:updateGradInput(input,gradOutput) 
end
function M:__tostring__() 
    return self.name .. '\n' .. self.model:__tostring() 
end
