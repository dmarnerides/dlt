
local dlt = require('dlt._env')
local M,parent = torch.class('dlt.Mnist','dlt.Loader',dlt)

-- Loads images and labels for training and validation sets in memory 
-- (byte [0-255]).
-- s.path 
-- s.assignPoint = function(point,iBatchMember,img,cls)
-- s.shuffle (defaults to true)
-- s.transform is a function of the whole dataset 
--    (since it's small and loaded once in memory)
-- s.download [true] downloads from 
--      https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
function M:__init(s)
    self.name = 'MNIST'    
    parent.__init(self,s)
    local train = paths.concat(s.path, 'train_32x32.t7')
    local val = paths.concat(s.path, 'test_32x32.t7')
    self.path = {training = train, validation = val}
    if s.download == nil then s.download = true end
    if s.download then
        if not paths.filep(self.path.training) 
            or not paths.filep(self.path.validation) then 
            os.execute('wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz' 
                            .. ' --directory-prefix=' .. s.path)
            os.execute('cd ' .. s.path .. '\n tar -xvzf mnist.t7.tgz \n ' 
                                .. ' mv mnist.t7/* . \n rm -r mnist.t7*' )
        end
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

function M:initInstance(setName)
    local f = torch.load(self.path[setName], 'ascii')
    self.set = self.set or {}
    self.set[setName] = {
            images = self.transform(f.data),
            labels = f.labels,
            nPoints = f.labels:size(1)
    }
end

function M:dataPoint(index,setName) 
    return self.set[setName].images[index],
           self.set[setName].labels[index] 
end