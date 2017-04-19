# Slurm

## Usage
Create scripts for the [slurm scheduler](https://slurm.schedmd.com/).

## Example
File: slurm.lua
```lua
local dlt = require('dlt')
-- Create object (remember, settings are automatically the ones from arg)
local slurm = dlt.Slurm()
local file,script = slurm:createScript('~') -- This will create the script in home directory
print('Slurm script location: ' .. file)
print('Contents:')
print(script)
```
File: precommands.sh
```bash
# These are the precommands
module load cudnn
```

Possible run:
```bash
## To run 'th something.lua' after runing the contents of precommands.sh, requested time 12:05:20
## Request one node with 8 tasks, on fat partition (but not fat2 because it's faulty)
th slurm.lua -sJobname myjob -sTh /path/to/something.lua -sPrecommands /path/to/precommands.sh -sTime 12:05:20 -sPartition fat -sExclude fat2 -sTasks 8 -sNodes 1
```
