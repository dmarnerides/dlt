local dlt = require('dlt._env')
local C,parent = torch.class('dlt.Cifar','dlt.Loader',dlt)

-- Loads images and labels for training and validation sets in memory 
-- (byte [0-255]).
------ Settings
-- Required:
-- s.path (Must contain cifar10-train.t7, cifar10-test.t7   
--                          or cifar100-train.t7, cifar100-test.t7)
-- Optional
-- s.assignPoint = function(point,iBatchMember,img,cls)
-- s.shuffle (defaults to true)
-- s.transform is a function of the whole dataset 
--      (since it's small and loaded once in memory)
-- s.name (100, 10) defaults to 10
-- s.download [true] if true and path does not contain datasets then will use 
--   https://github.com/soumith/cifar.torch to get the data

local function download(name,path)
    local fname = name .. 'BinToTensor.lua'
    local code  = 
        'https://raw.githubusercontent.com/soumith/cifar.torch/master/' 
         .. fname
    os.execute('wget ' .. code .. ' --directory-prefix=' .. path)
    os.execute('cd ' .. path .. '\n th ' .. fname)
end

function C:__init(s)
    
    if dlt.help.inTable({'10',10,'cifar10','Cifar10'},s.name) 
                or s.name == nil then
        self.name = 'cifar10'
    elseif dlt.help.inTable({'100',100,'cifar100','Cifar100'},s.name) then
        self.name = 'cifar100'
    else dlt.log:error('Unknown places name: ' .. s.name) end
    parent.__init(self,s)
    local train = paths.concat(s.path, self.name .. '-train.t7')
    local val = paths.concat(s.path, self.name .. '-test.t7')
    self.path = {training = train, validation = val}
    
    if s.download and (not paths.filep(self.path.training) 
            or not paths.filep(self.path.validation)) then
        download('C' .. self.name:sub(2,-1),s.path)
    end
    if not paths.filep(self.path.training) then   
        dlt.log:error('Could not find ' .. self.path.training) 
    end
    if not paths.filep(self.path.validation) then 
        dlt.log:error('Could not find ' .. self.path.validation) 
    end
    -- Transformation
    self.transform = s.transform or function(imagesTensor) 
                                        return imagesTensor 
                                    end
    -- Internals
    self.sets.testing = nil
end

function C:initInstance(setName)
    local f = torch.load(self.path[setName])
    self.set = self.set or {}
    self.set[setName] = {
            images = self.transform(f.data),
            labels = f.label,
            nPoints = f.label:size(1)
    }
end

function C:dataPoint(index,setName) 
    return  self.set[setName].images[index],
             self.set[setName].labels[index] 
end