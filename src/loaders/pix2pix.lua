-- All data from https://github.com/phillipi/pix2pix
local dlt = require('dlt._env')
local P,parent = torch.class('dlt.Pix2pix','dlt.Loader',dlt)

-- Loads images and labels for training and validation sets in memory (byte [0-255]).
------ Settings
-- Required:
-- s.path
-- Optional
-- s.assignPoint = function(point,iBatchMember,img,cls)
-- s.shuffle (defaults to true)
-- s.name (cityscapes,maps,facades,edges2handbags,edges2shoes) defaults to cityscapes
function P:__init(s)    
    if dlt.help.inTable({'cityscapes','maps','facades','edges2handbags','edges2shoes'},s.name) then
        self.name = s.name
    elseif s.name == nil then
        self.name = 'cityscapes'
    else dlt.log:error('Unknown places name: ' .. s.name) end
    parent.__init(self,s)
    s.path = paths.concat(s.path,self.name)
    if not paths.dirp(s.path) then dlt.log:error('Could not find: ' .. s.path) end

    -- Internals
    self.sizes = {
        cityscapes = {training = 2975, validation = 500},
        maps = {training = 1096, validation = 1098},
        facades = {training = 400, validation = 100, testing = 106},
        edges2handbags = {training = 138567, validation = 200},
        edges2shoes = {training = 49825, validation = 200}
    }

    if self.name ~= 'facades' then self.sets.testing = nil end
    -- paths
    self.path = {
        training = self.sets.training and paths.concat(s.path,'train') or nil,
        validation = self.sets.validation and paths.concat(s.path,'val') or nil,
        testing = self.sets.testing and paths.concat(s.path,'test') or nil
    }
    
    if self.name == 'edges2handbags' or self.name == 'edges2shoes' then self.format = '%d_AB.jpg'
    else self.format = '%d.jpg' end
end

function P:dataPoint(index,setName)
    return image.load(paths.concat(self.path[setName],string.format(self.format,index)),3,self.type)
end

function P:initInstance(setName)
    self.set = self.set or {}
    self.set[setName] = {nPoints = self.sizes[self.name][setName]}
end
