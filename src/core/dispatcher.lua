local dlt = require('dlt._env')
local D,parent = torch.class('dlt.Dispatcher',dlt)

function D:__init(experimentFunction,extras)
    -- Store experiment Function and parsed settings
    -- We do not store the actual experiment table
    --   but the function that returns it
    self.experiment = experimentFunction
    self.configuration = dlt.parse(nil,extras)
    -- First Make paths
    if not self.configuration.runRoot then 
        dlt.log:error('Dispatcher: runRoot not given.') 
    end
    if not self.configuration.experimentName then
        dlt.log:error('Dispatcher: experimentName not given.') 
    end
    self.configuration.runRoot = 
                            dlt.help.checkHomePath(self.configuration.runRoot)
    dlt.help.checkMakeDir(self.configuration.runRoot,'runRoot')
    self.configuration.runPath = paths.concat(self.configuration.runRoot,
                                            self.configuration.experimentName)
    dlt.help.checkMakeDir(self.configuration.runPath,'runPath')
    self.configuration.savePath = paths.concat(self.configuration.runPath,
                                                self.configuration.savePath)
    dlt.help.checkMakeDir(self.configuration.savePath)

    -- Create lua script that runs dispatcher inside run directory
    dlt.writeSettingsToFile(self.configuration,
                            paths.concat(self.configuration.runPath,
                                            'settings.txt'))
    torch.save(paths.concat(self.configuration.runPath,'dispatcher.t7'),self)
    local toRun = 
[[
local dlt = require('dlt')
local dispatcher = torch.load('dispatcher.t7')
dispatcher:runLoadedDispatcher()
]]
    local luaScriptPath = paths.concat(self.configuration.runPath,'script.lua')
    local file = io.open(luaScriptPath, 'w+')
    file:write(toRun)
    file:close()

end

function D:__call__(...) self:localRun(...) end

-- Do not use this, use D:run() if dispatcher was created 
--    and not loaded in script
function D:runLoadedDispatcher()
    -- Create settings
    -- We configure here because dispatcher may be created 
    --  on a machine without a gpu
    -- e.g. the login nodes of an HPC facility
    -- Set global configuration
    dlt.settings = self.configuration
    self.configuration = dlt.configure(self.configuration)
    -- Run experiment
    -- Default globals, tensor types, threads, seeds etc
    -- should be set at the beginning of the experiment function
    self.experiment()
end

function D:makeSlurm()
    self.batchScript = dlt.Slurm():createScript(self.configuration.runPath)
end

function D:run(preCommands,qlua)
    local launch = qlua and 'qlua' or 'th'
    os.execute('cd ' .. self.configuration.runPath .. '\n' .. preCommands .. 
                    '\n ' .. launch .. ' script.lua')
end

-- SLURM SUBMIT NEEDS FIXING
-- function D:submitSlurm()
--     slurm:submit(self.batchScript)
-- end