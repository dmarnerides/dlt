local dlt = require('dlt._env')

local H,parent = torch.class('dlt.HDR','dlt.Loader',dlt)

-- s.recursive
function H:__init(s)
    s.name = 'HDR'
    parent.__init(self,s)
    self.recursive = s.recursive
    self.path = s.path
    self.sets.validation = nil
    self.sets.testing = nil
end


function H:dataPoint(index,setName)
    setName = setName or self.currentSet
    return hdrimage.load(self.set[setName].list[index])
end

function H:initInstance(setName)
    self.set = self.set or {}
    setName = setName or self.currentSet
    local list =  dlt.help.getFiles(self.path,
                                      {hdr = true,exr = true},
                                      self.recursive)
    self.set[setName] = {
        list = list,
        nPoints = #list
    }
end
