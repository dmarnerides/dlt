local dlt = require('dlt._env')

local O,parent = torch.class('Optimizer',dlt)

-- Settings
-- [opt] = { [name = 'sgd', config = {}, hook = function(epoch,loss,current) ] }
-- [tensorType = 'float']
-- [useGPU = false]
-- [optimFile = nil]
function O:__init(opt,tensorType,useGPU,optimFile)
    opt = opt or {}

    self.tensorType = tensorType or 'float'
    self.useGPU = useGPU or false
    
    -- Set up optimizer, defaults to sgd
    if opt.name and optim[opt.name] == nil then dlt.log:error('Unknown optim type ' .. opt.name) end
    self.optim = opt.name and optim[opt.name] or optim['adam']
    -- Get optimizer state
    self.optimState = optimFile and torch.load(optimFile) or opt.config
    self.optimState = self.optimState or {} -- Make it an empty table if opt.config is nil
    -- Hook for updating the optimizer state
    self.optimHook = opt.hook or function(epoch,loss,current) return current end
    -- Transfer state to gpu (could have been loaded from checkpoint, saved on gpu)
    if self.useGPU then  nn.utils.recursiveType(self.optimState, 'torch.' .. dlt.help.tensorList.gpu[self.tensorType] ) end
  
end

function O:updateState(epoch,loss) 
    self.optimState = self.optimHook(epoch,loss,self.optimState) 
end

function O:save(filename)
    -- To save, first transfer to cpu and then recast
    if self.useGPU then  nn.utils.recursiveType(self.optimState, 'torch.FloatTensor') end 
    torch.save(filename, self.optimState)
    if self.useGPU then  nn.utils.recursiveType(self.optimState, 'torch.' .. dlt.help.tensorList.gpu[self.tensorType] ) end
end

-- Sets all numbers/tensors in optimizer state to defaults or zero
function O:resetState(defaults)
    for key,val in pairs(self.optimState) do
        if defaults[key] then self.optimState[key] = defaults[key] 
        else
            if torch.type(self.optimState[key]) == 'number' then  self.optimState[key] = 0 
            else if self.optimState[key].zero then self.optimState[key]:zero() end
            end
        end
    end
end

function O:step(f,param) self.optim(f,param,self.optimState) end