local dlt = require('dlt._env')

local T,parent = torch.class('Trainlog',dlt)

--
function T:__init(name,savePath) 
    -- Configure arguments
    if name == nil then dlt.log:error('Name not provided for trainlog.') end
    savePath = savePath or paths.concat()
    savePath = dlt.help.checkHomePath(savePath)
    dlt.help.checkMakeDir(savePath)
    -- Formats 
    self.numFormat = '%1.4e'
    self.percFormat = '%.1f'
    self.delim = ',\t'

    -- Make file
    self.filename = paths.concat(savePath,name .. '.log')
    local exists = paths.filep(self.filename)
    self.file = io.open(self.filename,'a')
    if not exists then self.file:write('loss\n') end
end

function T:log(loss)   
    self.file:write(string.format(self.numFormat,loss) .. '\n')
    self.file:flush()
end
