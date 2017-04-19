# Dispatcher

## Usage
```lua
dispatcher = dlt.Dispatcher(experimentFunction [,extras])
```
* `experimentFunction` is a function that runs an experiment.
* `extras` extra settings to parse. Must be provided if extra arguments were parsed already.

Useful for creating self-contained directories of a pre-configured experiment, (with slurm scheduler script ready to submit if required).

Need to provide with `experimentName` and `runRoot` settings. The experiment will be created/run in *runRoot/experimentName* .

## Example

File dispatch.lua
```lua
local dlt = require('dlt')
-- Can easily add extra settings to parse
local extras = {{'-localRun','true','Whether we run or create slurm script'},
                {'-dataPath','none','Path for data'}}

-- Get settings to use in closure
local s = dlt.parse(nil,extras)

-- Dispatcher needs a function that runs an experiment (could be doing anything really)
local function experiment()
    -- MUST get local reference to dlt
    local dlt = require('dlt')
    torch.setdefaulttensortype('torch.FloatTensor')
    -- Make experiment table
    local exp =  { model = { create = dlt.models.lenet5 }, 
             loader = dlt.Mnist{ path = s.dataPath, 
                                 transform = function(images) return images:float():div(255) end,
                                 assignPoint = function(batch,i,img,cls) 
                                                    batch.input[i]:copy(img)
                                                    batch.output[i] = cls
                                                end
                                },
             pointSize = {input = {1,32,32}, output = {}}, 
             criterion = nn.CrossEntropyCriterion()
        }
    -- Run trainer with given experiment
    dlt.Trainer(exp):run()
end

-- Create the dispatcher (MUST pass the extra settings in this case)
local dispatcher = dlt.Dispatcher(experiment,extras)
-- Run on local machine or make slurm script if we are on HPC machine
if s.localRun then 
    -- will add this string before running the script
    dispatcher:run('export THC_CACHING_ALLOCATOR=1')
    -- If we need to use qlua (e.g. for image.display()) then use 
    -- dispatcher:run('export THC_CACHING_ALLOCATOR=1',true)
else 
    -- Will only make the slurm script in 'runRoot/experimentName/job'
    -- Remember to pass slurm script arguments when invoking this
    dispatcher:makeSlurm() 
end
```

Possible run:
```bash
th dispatch.lua -runRoot ~/results -experimentName dispatcherTest -nGPU 1 -defGPU 2 -dataPath ~/data/mnist -batchSize 1000
```