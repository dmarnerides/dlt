local dlt = require('dlt._env')

local P,parent = torch.class('dlt.Places','dlt.Loader',dlt)


-- Settings
-- s.path
-- s.shuffle (defaults to true)
-- s.assignPoint = function(point,iBatchMember,img,cls)
-- s.type [byte]
-- s.name (365 or 205) defaults to 365

-- NOTE: Class is indexed from 0 so add 1 for criterion
function P:__init(s)
    if not dlt.have.csvigo then dlt.log:error('Places dataset requires csvigo package') end
    if dlt.help.inTable({'365',365,'2',2,'places2','Places2','places365','Places365'},s.name) or s.name == nil then
        self.name = 'places365'
    elseif dlt.help.inTable({'205',205,'1',1,'places','Places','places1','Places1','places205','Places205'},s.name) or not s.name then
        self.name = 'places205'
    else dlt.log:error('Unknown places name: ' .. s.name) end
    parent.__init(self,s)
    self.path = {}
    if self.name == 'places365' then
        
        for _,val in ipairs{'training','validation','testing'} do 
            self.path[val] = paths.concat(s.path,val)
            if not paths.dirp(self.path[val]) then  dlt.log:warning('Could not find '.. val .. ' path for ' .. self.name .. '. ' .. self.path[val]) end
        end
        
        self.fileList = s.fileList 
                    or { training = paths.concat(self.path.training,'places365_train_standard.txt'),
                        validation = paths.concat(self.path.validation,'places365_val.txt'),
                        testing = paths.concat(self.path.testing,'places365_test.txt')}
    else
        if not s.path then dlt.log:error('Path not provided for places205 loader.') end 
        if not paths.dirp(s.path) then dlt.log:error('Path provided for places205 loader does not exist. ' .. s.path) end
        
        for _,val in ipairs{'training','validation'} do self.path[val] = s.path end
        
        self.fileList = s.fileList 
                    or { training = paths.concat(self.path.training,'train_places205.csv'),
                        validation = paths.concat(self.path.validation,'val_places205.csv')}
        self.sets.testing = nil
    end
   
    self.type = s.type or 'byte'
end

-- Classes in places are 0 indexed, so we add 1.
function P:splitNameClass(index,setName)
    setName = setName or self.currentSet
    local imgName,cls = string.match(self.set[setName].list[index][1],"^/*([^%s]+)%s*(%d*)")
    return imgName,tonumber(cls) + 1
end

-- Potentially needs a faster implementation if to be used a lot
-- Only tested with 365
-- Made it to (quickly) look at some sample images
function P:sample(cls,setName)
    setName = setName or self.currentSet
    local index
    local nElem = self:size()
    while true do
        index = torch.random(1,nElem)
        local imgName,clsTest = self:splitNameClass(index,setName)
        if clsTest == cls then break end
    end
    local img,clsTest =  self:dataPoint(index,setName)
    return img
end

function P:getFullPathAndClass(index,setName)
    setName = setName or self.currentSet
    local imgName,cls = self:splitNameClass(index,setName)
    return paths.concat(self.path[setName],imgName), cls
end

function P:dataPoint(index,setName)
    setName = setName or self.currentSet
    local imgName,cls = self:splitNameClass(index,setName)
    return image.load(paths.concat(self.path[setName],imgName),nil,self.type), cls
end

function P:initInstance(setName)
    self.set = self.set or {}
    setName = setName or self.currentSet
    local list = csvigo.load({path = self.fileList[setName] , mode = 'large', verbose = false})
    self.set[setName] = {
        list = list,
        nPoints = #list
    }
end
