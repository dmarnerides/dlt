local dlt = require('dlt._env')

local D,parent = torch.class('dlt.Donkey',dlt)

function D:__init(loader,pointSize,batchSize,useLocks, garbageCollect, 
                    tensorType,set)
    self.loader = loader
    if loader == nil then 
        dlt.log:error('No loader provided for donkey.') 
    end
    if pointSize == nil then 
        dlt.log:error('No pointSize provided for donkey.') 
    end
    self.useLocks = useLocks or false
    self.batchSize = batchSize or 1
    self.garbageCollect = garbageCollect or 0
    self.garbageCounter = 0
    tensorType = tensorType or 'float'
    -- Create batch in main memory
    self.batch = dlt.help.createBatch(self.batchSize, pointSize,
                                        tensorType, 'cpu')
    -- Initialize Data
    self.loader:init(set)
    -- Create Timer
    self.timer = torch.Timer()
end

function D:getBatch(iPoint)
    self.timer:reset()
    if self.garbageCollect > 0 then self:collectgarbage() end
    if self.useLocks then mutex:lock() end
    self.loader:assignBatch(self.batch,iPoint,self.batchSize)
    if self.useLocks then mutex:unlock() end 
    return self.batch, self.timer:time().real
end

function D:collectgarbage()
    if self.garbageCounter >= self.garbageCollect then 
        self.garbageCounter = 0
        collectgarbage() 
    else self.garbageCounter = self.garbageCounter + 1 end
end