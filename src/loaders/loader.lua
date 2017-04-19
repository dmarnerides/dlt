-- Abstract loader class
local dlt = require('dlt._env')
local M = torch.class('dlt.Loader',dlt)

function M:__init(s)
    -- Defaults
    self.assignPoint = s.assignPoint or function() end
    self.shuffle = s.shuffle == nil and true or s.shuffle
    if torch.type(self.shuffle) ~= 'boolean' then 
        dlt.log:warning('shuffle must be boolean, setting to default (true).') 
        self.shuffle = true
    end
    -- Paths
    self.name = self.name or 'data'
    if not s.path then dlt.log:error('Path not provided for ' .. self.name .. ' loader.') end 
    s.path = dlt.help.checkHomePath(s.path)
    s.path = paths.concat(s.path)
    if not paths.dirp(s.path) then dlt.log:error('Path provided for ' .. self.name .. ' loader does not exist. ' .. s.path) end
    -- Internals
    self.shuffler = {}
    self.currentSet = 'training'
    self.sets = {training = true, validation = true, testing = true}
end

function M:transformIndex(index)  
    return self.shuffler[self.currentSet][(index - 1) % self.set[self.currentSet].nPoints + 1]   
end

function M:size(set) 
    set = set or self.currentSet
    return self.set[set].nPoints 
end

function M:mode(setName)
    if self.sets[setName] then  self.currentSet = setName
    else dlt.log:warning('Unknown mode: ' .. setName ..'. Keeping: ' .. self.currentSet ..  '.') end
end
function M:reshuffle() 
    self.shuffler[self.currentSet] = self.shuffle and torch.randperm(self.set[self.currentSet].nPoints):long() 
                            or self.shuffler[self.currentSet]
end
function M:get(index) return self:dataPoint(self:transformIndex(index),self.currentSet) end

function M:assignBatch(batch,iDataPoint,nPoints) 
    for i = iDataPoint,iDataPoint+nPoints - 1 do 
        local iPnt = (i - 1) % self.set[self.currentSet].nPoints + 1
        self.assignPoint(batch,i - iDataPoint + 1,self:get(iPnt)) 
    end
end

local function initHelp(self,setName)
     if not self.initialized[setName] then 
        self:initInstance(setName) 
        if self.shuffle then
            self.shuffler[setName] = torch.randperm(self.set[setName].nPoints):int() 
        else
            self.shuffler[setName] = {}  
            setmetatable(self.shuffler[setName],{__index = function(_,key) return key end})
        end
        self.initialized[setName] = true 
    end
end

function M:init(set)
    self.initialized = self.initialized or {}
    if set then  
        initHelp(self,set)
        self:mode(set)
    else 
        for setName,_ in pairs(self.sets) do 
            initHelp(self,setName) 
        end 
        self:mode('training')
    end
    return self
end