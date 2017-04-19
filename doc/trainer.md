# Trainer

## Usage
```lua
trainer = dlt.Trainer(experiment)
```
* `experiment` Table with experiment configuration
* Main functionality is `trainer:run()`.
* Automatically creates loss logs, saves model and optimizer configuration.
* Upon resume, automatically continues from previous checkpoint.
* Checkpoints every epoch AND according to `checkpointCondition`.

## Experiment Configuration

### `loader`
A [dlt.Loader](loader.md) (Without `loader:init()` called before).
```lua
experiment.loader = dlt.Mnist{path = '~/data/mnist'}
```
### `pointSize`
Table that describes each data point.
```lua
-- For class predictions use empty table {}
experiment.pointSize = {input = {1,32,32}, output = {}}
```

### `trainingType`
String. Currently supports:

* `'simple'`: Iterates training set, minimizes one model given a criterion. 
* `'validate'`: Same as simple but goes through validation set too.
* `'GAN'`: Generative adversarial networks (training set only)
* `'WGAN'`: Wasserstein GAN (training set only)
* `'BEGAN'`: Boundary Equilibrium GAN (training set only)

**NOTE:** `'simple'` and `'validate'`  assume that dataPoints have fields *input* and *output* while
`'GAN'`,`'WGAN'`,`'BEGAN'` assume fields *input* (z goes into generator), *sample* (x goes into discriminator), 
*output* (y out of discriminator)

### `model`
Table. Contents depend on `trainingType`.

* For `'simple'` or `'validate'` provide [model.create](model.md) [ and model.name ].
* For `'GAN'`,`'WGAN'`,`'BEGAN'` provide tables model.generator and model.discriminator each with [.create](model.md) [ and .name ].

```lua
experiment.trainingType = 'simple'
experiment.model = {create = functionThatCreatesModel, name = 'myAwesomeModel'}
-- OR
experiment.trainingType = 'GAN'
experiment.model ={ generator = {create = makeGeneratorFunction, name = 'Generator'},
                 discriminator = {create = '~/savedModels/discriminatorOnDisk.t7', name = 'Discriminator'} }
```

### `criterion`
An nn.Criterion or anything with correctly defined :forward() :backward() and :type()
* For `'simple'` or `'validate'` provide just the criterion
* For `'GAN'`,`'WGAN'`,`'BEGAN'` provide a table with model.discriminator and criterion.generator (useful for fancy GAN losses).
```lua
experiment.trainingType = 'simple'
experiment.criterion = nn.MSECriterion()
-- OR
experiment.trainingType = 'GAN'
experiment.criterion = {discriminator = nn.CrossEntropyCriterion} -- generator Criterion might not be needed
```
Note that `'WGAN'` and `'BEGAN'` do not need a criterion.

### `optim`
* For `'simple'` or `'validate'` provide a table with a name (config and hook optional)
* For `'GAN'`,`'WGAN'`,`'BEGAN'` provide table with optim.discriminator and optim.generator tables
```lua
experiment.trainingType = 'simple'
experiment.optim = {name = 'adam', config = {beta1 = 0.5}
                    -- hook (if provided) must return updated state. 
                    -- Called before each optimizer update ('simple'/'validate' modes)
                    hook = function(epoch,loss,currentState) 
                            if epoch == 2 then  currentState.beta1 = 0.2 end -- Or something that is actually useful
                            return currentState
                     }
-- OR
experiment.trainingType = 'GAN'
experiment.optim = {discriminator = {name = 'rmsprop', config = {learningRate = 5e-5}},
                    generator = {name = 'rmsprop', config = {learningRate = 1e-5}}}
```

### `trainingHooks`
Hooks for `'GAN'`,`'WGAN'`,`'BEGAN'` training, e.g. `onGeneratorTrainBegin(state)`.

### `checkpointCondition`

Number or function(state)

* If number then represents checkpointing frequency in minutes.
* If function, must return true when a checkpoint is required, otherwise return false.
    * Takes one argument `checkpointCondition(state)`
    * `state` is the trainer. Care must be taken not to change the internal state.

### Miscellaneous
* `nDFew`,`nDMany`,`manyInitial`,`manyFrequency` to set training schedule for `GAN` and `WGAN` (from [WGAN paper](https://arxiv.org/abs/1701.07875))
* `clampMin` [-0.01] and `clampMax` [0.01] FOR `WGAN`
* `diversityRatio` [0.5], `ktVarInit` [0], `ktLearningRate` [0.001] and `loss` [nn.AbsCriterion()] for `BEGAN`

## Example 1

File lenet.lua :
```lua
local dlt = require('dlt')
-- Train LeNet on MNIST. Yes, LeNet. Again. On MNIST.
local experiment = {
    loader = dlt.Mnist{
                path = '~/data/mnist',
                assignPoint = function(batch,i,img,cls) 
                                batch.input[i]:copy(img) 
                                batch.output[i] = cls
                              end
            },
    model = { create = dlt.models.lenet5, name = 'Lenet5' }, 
    trainingType = 'validate',
    pointSize = {input = {1,32,32}, output = {}},
    criterion = nn.CrossEntropyCriterion(),
    optim = {name = 'adadelta'} -- If not given it defaults to adam
}
dlt.Trainer(experiment):run()
```

Possible run:
```bash
# Run on 2 GPUs, load data only on master thread with batch size of 1000
# Save training log, models, optim state and checkpoint.t7 in ~/save/examples/1
th lenet.lua -nGPU 2 -nThreads 0 -batchSize 1000 -savePath ~/save/examples/1
```
## Example 2
File colornet.lua :
```lua
local dlt = require('dlt')
-- Train Colornet on Places2 (aka Places365).
-- Colornet paper: http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
local experiment = {
    loader = dlt.Places{
                path = '~/data/places365', type = 'float',
                assignPoint = function(batch,i,img,cls)
                    -- Scale to size used in paper
                    img = image.scale(img,224,224)
                    -- Get greyscale image
                    local grey = image.rgb2y(img)
                    -- Convert to Lab (also scaled to half width,height)
                    local lab = image.rgb2lab(image.scale(img,112,112))
                    lab:add(108):div(208) -- normalize ab to [0,1]
                    -- Model takes table input with two images (first is size invariant)
                    batch.input[1][i]:copy(grey)
                    batch.input[2][i]:copy(grey) 
                    -- Output is table with ab predictions and class
                    batch.output[1][i]:copy(lab[{{2,3},{},{}}])
                    batch.output[2][i] = cls
                end
            },
    model = { create = function() return dlt.models.colornet(224,224,365,1,2) end, name = 'Colornet' }, 
    trainingType = 'simple',
    -- Pointsize supports tables!
    pointSize = {  input =  { {1,224,224} , {1,224,224} } , output = { {2,112,112}, {} } },
    criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.CrossEntropyCriterion(),1/300),
    optim = {name = 'adam'}
}
dlt.Trainer(experiment):run()
```

Possible run:
```bash
# Run on GPU no. 2, load data on 8 threads, batch size 4 (In the paper they used 128 on a single K80 core)
# Do not overwrite checkpoints (i.e. keep models and optim states from each checkpoint)
# print timings 
th colornet.lua -nGPU 1 -defGPU 2 -nThreads 8 -batchSize 4 -saveAll true -verbose 5
```
## Example 3
File dcgan.lua :
```lua
local dlt = require('dlt')
-- Train DCGAN on MNIST
-- Code adapted from https://github.com/soumith/dcgan.torch
-- First define model creation (with weight init etc)
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end
local SBN, SConv, SFConv = nn.SpatialBatchNormalization,  nn.SpatialConvolution, nn.SpatialFullConvolution
local nc,nz,ndf,ngf = 3, 100, 64, 64
local function makeGenerator()
    local netG = nn.Sequential()
    netG:add(nn.View(nz,1,1))
    netG:add(SFConv(nz, ngf * 8, 4, 4)):add(SBN(ngf * 8)):add(nn.ReLU(true)) -- state size: (ngf*8) x 4 x 4
        :add(SFConv(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1)):add(SBN(ngf * 4)):add(nn.ReLU(true)) -- state size: (ngf*4) x 8 x 8
        :add(SFConv(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1)):add(SBN(ngf * 2)):add(nn.ReLU(true)) -- state size: (ngf*2) x 16 x 16
        :add(SFConv(ngf * 2, ngf, 4, 4, 2, 2, 1, 1)):add(SBN(ngf)):add(nn.ReLU(true)) -- state size: (ngf) x 32 x 32
        :add(SFConv(ngf , nc, 4, 4, 2, 2, 1, 1)):add(nn.Tanh()) -- state size: (nc) x 64 x 64
    netG:apply(weights_init)
    return netG
end
local function makeDiscriminator()
    local netD = nn.Sequential() -- input is (nc) x 64 x 64
    netD:add(SConv(nc, ndf, 4, 4, 2, 2, 1, 1)):add(nn.LeakyReLU(0.2, true)) -- state size: (ndf) x 32 x 32
        :add(SConv(ndf, ndf * 2, 4, 4, 2, 2, 1, 1)):add(SBN(ndf * 2)):add(nn.LeakyReLU(0.2, true)) -- state size: (ndf*2) x 16 x 16
        :add(SConv(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1)):add(SBN(ndf * 4)):add(nn.LeakyReLU(0.2, true)) -- state size: (ndf*4) x 8 x 8
        :add(SConv(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1)):add(SBN(ndf * 8)):add(nn.LeakyReLU(0.2, true)) -- state size: (ndf*4) x 4 x 4
        :add(SConv(ndf * 8, 1, 4, 4)):add(nn.Sigmoid()) -- state size: 1 x 1 x 1
        :add(nn.View(1):setNumInputDims(3)) -- state size: 1
    netD:apply(weights_init)
    return netD
end

-- Define and run experiment
local experiment = {
    loader = dlt.CelebA{ 
                    path = '~/data/celeba', 
                    type = 'float',
                    assignPoint = function(batch,i,img,cls) 
                        img = image.scale(image.crop(img,'c',178,178),64,64) -- crop and resize
                        batch.sample[i]:copy(img:mul(2):add(-1)) 
                    end
        },
    trainingHooks = { getGeneratorInput = function(batch) return batch.input:normal(0,1)  end },
    model = { discriminator = { create = makeDiscriminator, name = 'Discriminator' },
                 generator = { create = makeGenerator, name = 'Generator' }},
    trainingType = 'GAN',
    checkpointCondition = 1, -- Checkpoint every minute
    pointSize = {  input =  {nz} , sample = {nc,64,64}, output = {} },
    criterion = {discriminator = nn.BCECriterion()},
    optim = {discriminator = {name = 'adam', config = {learningRate = 2e-4, beta1 = 0.5}},
                generator = { name = 'adam', config = {learningRate = 2e-4, beta1 = 0.5} } }
}
dlt.Trainer(experiment):run()
```

Possible run:
```bash
# Run on GPU, load data on 4 threads, batch size 8
# Do not overwrite checkpoints (i.e. keep models and optim states from each checkpoint)
th dcgan.lua -nGPU 1  -nThreads 4 -batchSize 8 -saveAll true
```