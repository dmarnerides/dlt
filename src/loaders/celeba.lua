local dlt = require('dlt._env')

local P,parent = torch.class('dlt.CelebA','dlt.Loader',dlt)

-- Data: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
-- Settings
-- s.path
-- s.shuffle (defaults to true)
-- s.assignPoint = function(point,iBatchMember,img)
-- s.type [byte]
function P:__init(s)
    self.name = 'CelebA'
    parent.__init(self,s)
    self.nAll = 202599
    self.path = s.path
    self.type = s.type or 'byte'
    self.sets.testing = nil
    self.sets.validation = nil
end

function P:dataPoint(index,setName)
    local file = paths.concat(self.path,string.format('%06d.jpg',index))
    return image.load(file,nil,self.type)
end

function P:initInstance(setName)
    self.set = self.set or {}
    self.set[setName] = { nPoints = self.nAll }
end