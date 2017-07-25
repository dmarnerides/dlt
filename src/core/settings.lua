local dlt = require('dlt._env')

-- Helps with parsing arguments and setting defaults
function dlt.parse(out,extra,onlyExtra)
    if dlt.settings == nil then
        local seed = torch.random()
        local cmd = torch.CmdLine()
        -- First print extra settings     
        if extra then 
            cmd:text('User provided settings:')
            for _,val in ipairs(extra) do cmd:option(unpack(val)) end 
        end
        cmd:text()
        if not onlyExtra then
            cmd:text('Global(ish) Settings:')
            cmd:option('-verbose',        3,           'Verbose level.')
            cmd:option('-makeLogFile',    'false',     'Whether log output is saved to file.')
            cmd:option('-defGPU',         1,           'Default GPU.')
            cmd:option('-tensorType',     'float',     'Tensor Type for model and optimizer. ')
            cmd:option('-batchSize',      128,         'Batch size.')
            cmd:option('-maxEpochs',      1000,        'Maximum number of epochs.')
            cmd:text()
            cmd:text('Data:')
            cmd:option('-useLocks',       'false',     'Whether to use locks before and after loading data in threads.')
            cmd:option('-collectGarbage', 50,          'Garbage collection frequency (per iteration) for each loader thread.')
            cmd:option('-nGPU',           0,           'Number of gpus.')
            cmd:option('-nThreads',       0,           'Number of threads.')
            cmd:option('-seed',           seed,        'Seed (default is random).')
            cmd:option('-epochReshuffle', 'true',      'Whether to reshuffle every epoch.')
            cmd:text()
            cmd:text('Model:')
            cmd:option('-useCudnn',       'true',      'Whether to use cudnn.')
            cmd:option('-cudnnFastest',   'true',      'Whether to use cudnn Fastest.')
            cmd:option('-cudnnBenchmark', 'true',      'Whether to use cudnn Benchmark.')
            cmd:option('-cudnnVerbose',   'false',     'Whether cudnn is verbose.')
            cmd:option('-dptFlatten',     'true',      'Whether to use DPT flattenParameters.')
            cmd:option('-dptNccl',        'false',     'Whether to use DPT NCCL.')
            cmd:text()
            cmd:text('Trainer:')
            cmd:option('-savePath',       'save',      'Directory name for saved progress.')
            cmd:option('-saveAll',        'false',     'Whether to keep all saved checkpoints.')
            cmd:text()
            cmd:text('Dispatcher:')
            cmd:option('-experimentName', 'experiment', 'Name of experiment.')
            cmd:option('-runRoot',        'runRoot',    'Root path for runs.')
            cmd:text()
            cmd:text('Slurm:')
            cmd:option('-sTime',          '48:00:00',  'Requested time hh:mm:ss.')
            cmd:option('-sNodes',          1,          'Nodes to request.')
            cmd:option('-sTasks',          1,          'Tasks to request.')
            cmd:option('-sPartition',     'gpu',       'Partition on cluster.')
            cmd:option('-sMempercpu',      32240,      'Memory per task.')
            cmd:option('-sMem',            62112,      'Total Memory.')
            cmd:option('-sGres',          'none',      'Generic resource to request.')
            cmd:option('-sExclude',       'none',      'Nodes to exclude.')
            cmd:option('-sRequest',       'none',      'Nodes to request.')
            cmd:option('-sJobname',       'job',       'Name of job.')
            cmd:option('-sOutname',       'default',   'Name of output slurm file.')
            cmd:option('-sEmail',         'none',      'Email address for notifications.')
            cmd:option('-sPrecommands',   'none',      'Commands to run before main script.')
            cmd:option('-sTh',            'none',      'Torch script to run with full path.')
            cmd:text()
        end
        dlt.settings = cmd:parse(arg) 
        
        -- Handle booleans (convert strings 'true' 'false' to boolean)
        for _,val in ipairs{'useCudnn','cudnnFastest','cudnnBenchmark',
                            'cudnnVerbose','dptFlatten','dptNccl','saveAll', 
                            'useLocks', 'makeLogFile'} do
            if dlt.settings[val] ~= nil then
                dlt.settings[val] = dlt.settings[val] == 'true'
            end
        end 
        if extra then
            for _,val in ipairs(extra) do 
                if val[2] == 'false' or val[2] == 'true' then
                    dlt.settings[val[1]:sub(2,-1)] = 
                            dlt.settings[val[1]:sub(2,-1)] == 'true'
                end
            end 
        end
    end
    if out then for key,val in pairs(dlt.settings) do out[key] = val end end
    
    return dlt.settings
end

function dlt.configure(s)
    
    -- Set verbose level
    dlt.log:setLevel(s.verbose)

    -- Make log file
    if s.makeLogFile and not dlt.__setLoggerFile then 
        dlt.log:setFile(paths.concat(s.savePath,'log')) 
        dlt.__setLoggerFile = true
    end 

    -- Check GPU
    local availGPU = dlt.have.cutorch and cutorch.getDeviceCount() or 0
    if s.nGPU > availGPU then
        dlt.log:warning(string.format(
                        'Available GPUs are %d, setting nGPU to %d', 
                            availGPU,availGPU))
        s.nGPU = availGPU
    end

    -- Check cudnn
    if s.useCudnn and not dlt.have.cudnn then
        dlt.log:warning('Cudnn could not be loaded make sure it is' .. 
                        ' installed. Switching cudnn use off.')
        s.useCudnn = false
    end

    -- Set seeds
    torch.manualSeed(s.seed)
    if s.nGPU > 0  then cutorch.manualSeedAll(s.seed) end
    
    -- Set default GPU
    if s.nGPU > 0  then 
        if s.nGPU > 1 and s.defGPU ~= 1 then
            dlt.log:warning('For multi-GPU use GPU 1 as default. ' ..
                            'Setting defGPU = 1.')
            s.defGPU = 1
        end
        cutorch.setDevice(s.defGPU) 
    end 
    return s
end

function dlt.reportExperiment(settings)
    dlt.log:section('Settings for ' .. settings.experimentName)
    dlt.log:yell('Verbose level: ' .. settings.verbose)
    dlt.log:yell('Seed: ' .. settings.seed)
    dlt.log:yell('Max Epochs: ' .. settings.maxEpochs)
    if settings.nGPU > 0  then
        dlt.log:yell('Running on GPUs (' .. settings.nGPU .. ')')
        dlt.help.logAllGPUMemory(settings.nGPU)
        if settings.useCudnn then 
            dlt.log:yell(string.format('Using cudnn with: fastest = %s,'
                                    ..' benchmark = %s, verbose = %s',
                                    tostring(settings.cudnnFastest),
                                    tostring(settings.cudnnBenchmark),
                                    tostring(settings.cudnnVerbose) ) )
        else
            dlt.log:yell('Not using cudnn')
        end
        if settings.nGPU > 1 then
            dlt.log:yell(string.format(
                            'GPU parallelism using DataParallelTable with:'
                            .. ' flattenParameters = %s, NCCL = %s',
                            tostring(settings.dptFlatten),
                            tostring(settings.dptNccl)))
        end
    else 
        dlt.log:yell('Running on CPU')
    end
    
    if settings.saveAll then 
        dlt.log:yell('Will save all checkpoints') 
    else 
        dlt.log:yell('Will only save latest checkpoint') 
    end
    dlt.log:yell('Save path: ' .. settings.savePath)
    dlt.log:yell('Mini-batch size: ' .. settings.batchSize)
    if settings.nThreads == 1 then 
        dlt.log:yell('Will be loading data using 1 thread.') 
    elseif settings.nThreads > 1 then 
        dlt.log:yell('Will be loading data using ' .. settings.nThreads 
                                                            .. ' threads.')
    else 
        dlt.log:yell('Will not be using threads to load data.') 
    end
    if settings.useLocks then 
        dlt.log:yell('Locks are turned on.') 
    else 
        dlt.log:yell('Locks are turned off.') 
    end
    dlt.log:endSection()
end

function dlt.writeSettingsToFile(s,fileName)
    local file = io.open(fileName, 'w+')
    for set,val in pairs(s) do 
        file:write('-' .. set .. ' ' .. tostring(val) .. '\n') 
    end
    file:close()
end