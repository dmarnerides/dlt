local dlt = require('dlt._env')

local M,parent = torch.class('dlt.Model',dlt)


function M:__init(modelCreate,name)
    self.name = name or 'model'
    if torch.type(self.name) ~= 'string' then dlt.log:error('Model name must be a string.') end
    self.name = self.name:gsub("^%l", string.upper) 
    dlt.log:section(self.name .. ' initialization')    
    dlt.parse(self)
    dlt.configure(self)
    if torch.type(modelCreate) == 'string' then
        modelCreate = dlt.help.checkHomePath(modelCreate)
        if not paths.filep(modelCreate) then dlt.log:error('Could not find model ' .. modelCreate) end
        self.model = torch.load(modelCreate)
        dlt.log:yell('Loaded ' .. self.name ..   ' from file.');
    elseif torch.type(modelCreate) == 'function' then
        self.model = modelCreate()
        dlt.log:yell('Created ' .. self.name ..   '.') 
    else dlt.log:error('dlt.Model parameter (modelCreate) must be a string or a function.') end

    self:processModel()

    dlt.log:yell(string.format(self.name .. ' parameters: %d.',self.parameters:size(1)))
    collectgarbage()
    dlt.log:endSection()
end

function M:getCleanModel()
    -- Clear State
    self.model:clearState()
    -- Remove DataParallelTable
    self.model = torch.type(self.model) == 'nn.DataParallelTable' and self.model:get(1) or self.model
    -- Remove cudnn
    if self.useCudnn then 
        cudnn.convert(self.model,nn) 
        cutorch.synchronizeAll()
    end
    -- Return CPU model
    return self.model:type('torch.' .. dlt.help.tensorList.cpu[self.tensorType])
end

function M:processModel()

    -- 1. Have simple CPU model
    -- 2. Configure CPU tensor types
    -- 3. Convert to cudnn (modules)
    -- 4. Call model:type('TensorType')
    -- 6. Add to dpt
    -- 7. Launch threads on dpt -- Some problems with checkpointing, commented out atm

    -- First cast to cpu
    self.model:type('torch.' .. dlt.help.tensorList.cpu[self.tensorType])

    if self.nGPU > 0 then 
        -- Convert to cudnn
        if self.useCudnn then
            cudnn.fastest, cudnn.benchmark, cudnn.verbose  = self.cudnnFastest,self.cudnnBenchmark,self.cudnnVerbose
            cudnn.convert(self.model,cudnn)
            self:sync()
        end

        -- Call :cuda()
        self.model:type('torch.' .. dlt.help.tensorList.gpu[self.tensorType])

        -- If multiple GPUs wrap in DataParallelTable
        -- Not sure if call to :threads is a good choice here
        if self.nGPU > 1 then
            local fastest, benchmark, verbose = self.cudnnFastest,self.cudnnBenchmark,self.cudnnVerbos
            local dpt = nn.DataParallelTable(1, self.dptFlatten,self.dptNccl)
                        :add(self.model, torch.range(1, self.nGPU):totable())
                        -- :threads(function()
                        --     local cudnn = require 'cudnn'
                        --     cudnn.fastest, cudnn.benchmark, cudnn.verbose = fastest, benchmark, verbose
                        -- end)
            self.model = dpt
            -- self.model.gradInput = nil
            -- self.model:type('torch.' .. dlt.help.tensorList.gpu[self.tensorType])
            self:sync()
        end
    end
    self:training()
    -- Flatten Paramters
    self.parameters, self.gradParameters = self.model:getParameters()
end

function M:sync() if self.nGPU > 0 then cutorch.synchronizeAll() end end

function M:save(filename)
    torch.save(filename, self:getCleanModel()) 
    self:processModel()
end

function M:training() self.model:training() end
function M:evaluate() self.model:evaluate() end
function M:zeroGradParameters() self.model:zeroGradParameters() end
function M:forward(input)  return self.model:forward(input) end
function M:backward(input,gradOutput) return self.model:backward(input,gradOutput) end
function M:updateGradInput(input,gradOutput) return self.model:updateGradInput(input,gradOutput) end
function M:__tostring__() return self.name .. '\n' .. self.model:__tostring() end
