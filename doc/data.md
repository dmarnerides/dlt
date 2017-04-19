# Data

## Usage
```lua
data = dlt.Data( loader, pointSize [, datasets, currentEpoch] )
```
* `loader` A data [loader](loader.md).
* `pointSize` Table of point elements and their sizes.
* `datasets` Table of datasets to use from loader.
* `currentEpoch` Useful for checkpointing and resuming runs.
* Main functionality is `data:iterate(callbacks)`.

## Example

File data.lua :
```lua
local dlt = require('dlt')
-- Use mnist loader as example
local mnist = dlt.Mnist{path = '~/data/mnist', shuffle = false,
                assignPoint = function(batch,i,img,cls) 
                                batch.img[i]:copy(img)
                                batch.cls[i] = cls
                            end,
                -- Mnist loads as ByteTensors, so we can use transform to convert all images to [0,1] floats.
                transform = function(images) return images:float():div(255) end}
-- input is 32x32 image with 1 channel. 
-- class is dimensionless (might need to use {1} instead of {} depending on criterion)
local pointSize = {img = {1,32,32}, cls = {}}

-- Create data iterator for training and validation
local data = dlt.Data( mnist, pointSize, {'training','validation'})

-- Make closure variables
local trainClassSum, valClassSum = 0,0
local trainCount, valCount = 0,0
local didCheckpoint = false
local batchSize, batchType
-- Iterate datasets with checkpointing and termination conditions
data:iterate{
    training = function(batch)
                    if didCheckpoint then return true, 'Did Checkpoint (Training)!' end -- return a termination statement
                    -- Here we have access to the batch loaded from the dataset
                    trainClassSum = trainClassSum + batch.cls:sum()
                    trainCount = trainCount + batch.cls:nElement()
                    batchSize = batch.img:size(1)
                    batchType = torch.type(batch.img)
                end,
    validation = function(batch)
                    if didCheckpoint then return true, 'Did Checkpoint (Validation)!' end -- return a termination statement
                    valClassSum = valClassSum + batch.cls:sum()
                    valCount = valCount + batch.cls:nElement()
                 end,
    checkpoint = function() -- This is called at the end of EVERY iteration
                    -- Stop at the first validation step
                    if valCount > 0 then  didCheckpoint = true end
                end
}
dlt.log:section('Results')
dlt.log:yell(string.format('Training Class Average for %d points: %.2f',trainCount,trainClassSum/trainCount ) )
dlt.log:yell(string.format('Validation Class Average for %d points: %.2f',valCount,valClassSum/valCount ) )
dlt.log:yell(string.format('Batch size %d, type %s',batchSize,batchType))
dlt.log:endSection()
```


Possible runs:
```bash
# Run on CPU only on master thread with batchSize of 16
th data.lua -nGPU 0 -nThreads 0 -batchSize 16
# Run on GPU no. 2 (callbacks batch will be on GPU 2) with 4 threads (loading of data)
# Note that for mnist this will not make much difference, (might actually be slower)
th data.lua -nGPU 1 -defGPU 2 -nThreads 4 
# Use verbose 5 to get timings printed to console
# Use a batch with double precision
th data.lua -verbose 5 -tensorType double
```