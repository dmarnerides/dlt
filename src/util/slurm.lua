local dlt = require('dlt._env')

local S,parent = torch.class('Slurm',dlt)

-- Class for slurm scheduler support
-- Initialize Slurm object with settings
function S:__init()
    dlt.parse(self)
    dlt.configure(self)
    self.sTh = dlt.help.checkHomePath(self.sTh)
    self.sPrecommands = dlt.help.checkHomePath(self.sPrecommands)
end
                  
function S:createScript(runPath)
    -- default runPath is jobname
    runPath = runPath or self.sJobname
    runPath = dlt.help.checkHomePath(runPath)
    -- Make script
    local script = [[#!/bin/bash]] .. '\n\n'

    -- Job Name, time, nodes, tasks, partition
    script = script .. [[#SBATCH --job-name=]] .. self.sJobname .. '\n'
    script = script .. [[#SBATCH --time=]] .. self.sTime .. '\n'
    script = script .. [[#SBATCH --nodes=]] .. self.sNodes .. '\n'
    script = script .. [[#SBATCH --ntasks-per-node=]] .. self.sTasks .. '\n'

    -- Memory. If total memory is given and we are not using fat nodes use that, otherwise request mem-per-cpu
    -- This is a hack for the HPC facility i'm currently using. Should fix
    if self.sMem ~= 0 and self.sPartition ~= 'fat' then
        script = script .. [[#SBATCH --mem=]] .. self.sMem .. '\n'
    else
        script = script .. [[#SBATCH --mem-per-cpu=]] .. self.sMempercpu .. '\n'
    end

    -- Partition
    if self.sPartition ~= 'none' then 
        script = script .. [[#SBATCH --partition=]] .. self.sPartition .. '\n'
    end

    -- Generic resources request
    if self.sGres ~= 'none' then
        script = script .. [[#SBATCH --gres=]] .. self.sGres .. '\n'
    end

    -- sExclude nodes
    if self.sExclude ~= 'none' then 
         script = script .. [[#SBATCH --sExclude=]] .. self.sExclude .. '\n'
    end

    -- Request nodes
    if self.sRequest ~= 'none' then 
         script = script .. [[#SBATCH --nodelist=]] .. self.sRequest .. '\n'
    end

    -- Output name 
    if self.sOutname == 'default' then
        local outputString = 'slurm_' .. self.sJobname .. [[_%A]]
        script = script .. [[#SBATCH --output=]] .. outputString .. '\n'
    else
        script = script .. [[#SBATCH --output=]] .. self.sOutname .. '\n'
    end

    -- email
    if self.sEmail ~= 'none' then
        script = script .. [[#SBATCH --mail-type=ALL]] .. '\n'
        script = script .. [[#SBATCH --mail-user=]] .. self.sEmail .. '\n\n'
    end
    script = script .. '\n'

    -- pre-commands
    if self.sPrecommands ~= 'none' then
        -- check pre-commands
        if not paths.filep(paths.concat(runPath,self.sPrecommands)) then
            dlt.log:error('Could not find file with pre-commands ' .. paths.concat(runPath,self.sPrecommands) )
        end
        local preFile = io.open(paths.concat(runPath,self.sPrecommands),'r')
        local commands = preFile:read('*all')
        preFile:close()
        script = script .. commands ..'\n\n' 
    end

    -- check script
    if self.sTh ~= 'none' and not paths.filep(paths.concat(runPath,self.sTh)) then
        dlt.log:error('Could not find torch script ' .. paths.concat(runPath,self.sTh) )
    end
    
    -- if runPath == nil then self.runPath = self.sJobname end
    if not paths.dirp(runPath) and not paths.mkdir(runPath) then
        dlt.log:error('Unable to create directory: ' .. runPath .. '\n')
    end

    -- cd to runPath in slurm script
    script = script .. 'cd ' .. paths.concat(runPath) .. '\n\n'
    -- invoke torch
    if self.sTh ~= 'none' then script = script .. 'th ' .. self.sTh .. '\n' end 

    -- Write slurm script to runPath
    local fullScriptName = paths.concat(runPath,'job')
    local file = io.open(fullScriptName, 'w+')
    file:write(script)
    file:close()
    
    return fullScriptName, script

end

-- This does not work properly, will have to double check. For now submit manually
function S:submit(scriptFile)
    if not paths.filep(scriptFile) then
        dlt.log:error('Could not find batch script ' .. scriptFile)
    end
    local scriptPath = paths.dirname(scriptFile)
    dlt.log:print('Submitting script')
    os.execute('cd ' .. scriptPath ..  '\n' .. 'sbatch ' .. scriptFile )
end