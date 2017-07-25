# Model

## Usage
```lua
net = dlt.Model(create [, name, save])
```
* `create` is a function that returns a network OR is the path (string) to a torch serialized network.
* `name` string, defaults to "model"
* `save` boolean, defaults to true. Whether model will be saved to disk.

## Example

File model.lua :
```lua
local dlt = require('dlt')
-- model creation function
local function myCustomModelFunction() 
    return nn.Sequential()
            :add(nn.Linear(10,10))
            :add(nn.Sigmoid()) 
end

-- Create dlt.Model instance
local net = dlt.Model(myCustomModelFunction,'customModel')

print(net)

-- Some standard functions are provided
net:evaluate()
net:training()
net:zeroGradParameters()

-- Make a test input mini batch of size 4 (useGPU can be useful)
local input = net.useGPU and torch.Tensor(4,10) or torch.Tensor(4,10):cuda() -- 
local gradOutput = input:clone()
local output = net:forward(input)
local gradInput = net:backward(input,gradOutput)

-- Save to file (If net is on gpu, it first takes it to RAM to save, then brings back to GPU automatically)
net:save('customModel.t7')
local output = net:forward(input)
local gradInput = net:backward(input,gradOutput)

-- Load saved model
local savedModel = dlt.Model('customModel.t7','savedModel')
print(savedModel)
print(net) -- net is completely different from savedModel

-- Load model without need of dlt (this will in RAM)
local noDLTmodel = torch.load('customModel.t7')
print(noDLTmodel)

```

Possible runs:
```bash
# Run on CPU
th model.lua -nGPU 0 
# Run on GPU no. 2
th model.lua -nGPU 1 -defGPU 2  
# Run on multiple GPUs (DataParallelTable) using cudnn 
th model.lua -nGPU 2 -useCudnn true
```
