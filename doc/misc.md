## Settings

```lua
[s] dlt.parse([out,extra])
```
* Passes arguments (arg) and returns table of settings for dlt.
* `out` is a table to place all settings in (useful for objects to pass self)
* `extra` is a table of extra settings that need parsing.

For the full list of available settings run:
```bash
th -e "require('dlt').parse()"
```

### Example
File: settings.lua
```lua
local dlt = require('dlt')

local myTable = { notAnArgSetting = 'yes' }

local s = dlt.parse(myTable,{ {'-check', 'false', 'Boolean setting.'},
                           {'-myNumber', 0, 'Number setting.'},
                           {'-myMessage', 'message', 'String setting.'}})


print(torch.type(s.check)) -- boolean

print('s --> ' .. s.myMessage .. ': ' .. s.myNumber)
print(s.notAnArgSetting) -- nil

print('myTable --> ' .. s.myMessage .. ': ' .. s.myNumber)
-- myTable still has all other settings (unless overwritten due to name clash)
print(myTable.notAnArgSetting) -- yes

-- Also parses dlt settings
print(s.nGPU)
print(myTable.batchSize)
```
Example run:
```bash
th settings.lua -myMessage "My Number is" -myNumber 42
```

## Optimizer

A thin wrapper to [optim](https://github.com/torch/optim).

Main functionality is for consistent saving of state when used in conjuction with [dlt.Trainer](trainer.md) (conversion of Tensors to/from GPU)

Default optimizer is [Adam](https://github.com/torch/optim/blob/master/adam.lua)

## Logger

When *dlt* loaded a logger is created `dlt.log`:
* Has 6 verbose levels:
    * [1-6]: error, warning, yell, say, detail, debug
    * Use `dlt.log:say('Something')`
    * If `dlt.log:error('message')` is used, execution is terminated
    * set: `dlt.log:setLevel(level)`
* Prints stuff in terminal friendly boxes
* Can create sections:
    * `dlt.log:section('My Section')`, `dlt.log:endSection()`

## Colorspaces

Supports colorspace conversions (not the ones from [image](https://github.com/torch/image)) for XYZ, IPT, LMS, Lαβ.

Example:
```lua
local dlt = require('dlt')
local img = image.load('image.jpg')
iptImage = dlt.color.rgb2ipt(img)
```