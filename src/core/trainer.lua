local dlt = require('dlt._env')

local T,parent = torch.class('dlt.Trainer',dlt)

function T:__init(experiment)
    dlt.parse(self)
    dlt.configure(self)
    dlt.reportExperiment(self)
    dlt.log:section('Trainer initialization')

    -- Generic settings
    self.hooks = experiment.trainingHooks or {}
    self.useGPU = self.nGPU > 0
    -- format
    self.format = self.nGPU > 0 and 'gpu' or 'cpu'
    self.format = 'torch.' .. dlt.help.tensorList[self.format][self.tensorType]
 
    -- Configure training type
    local trainingTypes = {'simple', 'validate', 'GAN', 'WGAN','BEGAN'}
    self.trainingType = experiment.trainingType or 'simple'
    if not dlt.help.inTable(trainingTypes,self.trainingType) then dlt.log:error('Unkown training type: ' .. self.trainingType) end
    self.datasets = self.trainingType == 'validate' and {'training','validation'} or {'training'}
    
    -- Load checkpoint
    self:loadCheckpoint()

    -- Create data
    self.data = dlt.Data(experiment.loader,experiment.pointSize,self.datasets,self.currentEpoch)

    -- Setup training
    if self.trainingType == 'simple' or self.trainingType == 'validate' then
        self.trainingCallback = self.standardCallback 
        local modelCreate = self.modelFile or experiment.model.create
        self.model = dlt.Model(modelCreate,experiment.model.name)
        self.optimizer = dlt.Optimizer(experiment.optim,self.tensorType,self.useGPU,self.optimFile)
        self.criterion = experiment.criterion
        self.criterion:type(self.format)
        self.trainlog = dlt.Trainlog('training',paths.concat(self.savePath))
        if self.trainingType == 'validate' then 
            self.validationlog = dlt.Trainlog('validation',paths.concat(self.savePath))
        end
    elseif dlt.help.inTable({'GAN','WGAN'},self.trainingType) then
        self.model, self.optimizer = {},{}
        -- Hooks and callbacks
        self.defaultHooks[self.trainingType](self,self.hooks)
        self.trainingCallback = self[self.trainingType .. 'Callback']
        -- Training frequencies
        self.nDSteps, self.nGMany = 0, 0
        defFreq = {nDFew = {GAN = 1, WGAN = 5}, 
                    nDMany = {GAN = 1, WGAN = 5},
                    manyInitial = {GAN = 1, WGAN = 25}, 
                    manyFrequency = {GAN = 1, WGAN = 100}}
        for _,val in ipairs{'nDFew','nDMany','manyInitial','manyFrequency'} do 
            self[val] = experiment[val] or defFreq[val][self.trainingType]
        end
        if self.trainingType == 'WGAN' then
            self.clampMin = experiment.clampMin or 0.01
            self.clampMax = experiment.clampMax or 0.01
        end 
        -- Create models,optimizers and criterions
        self.modelFile = self.modelFile or {}
        self.optimFile = self.optimFile or {}
        self.criterion = experiment.criterion
        for _,val in ipairs{'discriminator','generator'} do
            local modelCreate = self.modelFile[val] or experiment.model[val].create
            local modelName = experiment.model[val].name or val
            self.model[val] = dlt.Model(modelCreate,modelName)
            self.optimizer[val] = dlt.Optimizer(experiment.optim[val],self.tensorType,self.useGPU,self.optimFile[val])
            if self.criterion and self.criterion[val] then self.criterion[val]:type(self.format) end
        end
        
        self.discriminatorLog = dlt.Trainlog('discriminator',paths.concat(self.savePath))
        self.generatorLog = dlt.Trainlog('generator',paths.concat(self.savePath))
    elseif self.trainingType == 'BEGAN' then
        self.model, self.optimizer = {},{}
        self.defaultHooks[self.trainingType](self,self.hooks)
        self.trainingCallback = self.BEGANCallback
        self.modelFile = self.modelFile or {}
        self.optimFile = self.optimFile or {}
        for _,val in ipairs{'discriminator','generator'} do
            local modelCreate = self.modelFile[val] or experiment.model[val].create
            local modelName = experiment.model[val].name or val
            self.model[val] = dlt.Model(modelCreate,modelName)
            self.optimizer[val] = dlt.Optimizer(experiment.optim[val],self.tensorType,self.useGPU,self.optimFile[val])
        end

        self.diversityRatio = experiment.diversityRatio or 0.5
        self.ktVar = experiment.ktVarInit or 0
        self.ktLearningRate = experiment.ktLearningRate or 0.001
        self.loss = experiment.loss or nn.AbsCriterion()
        self.loss:type(self.format)

        self.discriminatorLog = dlt.Trainlog('discriminator',paths.concat(self.savePath))
        self.aeLog = dlt.Trainlog('autoencoder',paths.concat(self.savePath))
        self.generatorLog = dlt.Trainlog('generator',paths.concat(self.savePath))
        self.ktVarLog = dlt.Trainlog('kt',paths.concat(self.savePath))
        self.convergenceLog = dlt.Trainlog('convergence',paths.concat(self.savePath))

    end

    -- Replace default callbacks if given 
    self.trainingCallback = experiment.trainingCallback or self.trainingCallback
    
    -- Checkpoint condition
    self.checkpointCondition = experiment.checkpointCondition or function() return false end 
    
    if torch.type(self.checkpointCondition) == 'number' and self.checkpointCondition <=0 then
        dlt.log:error('Checkpoint Condition must be positive number (minutes) or function')
    end
    if torch.type(self.checkpointCondition) ~= 'function' and torch.type(self.checkpointCondition) ~= 'number' then 
            dlt.log:error('Checkpoint Condition must be positive number (minutes) or function')
    end
    if torch.type(self.checkpointCondition) == 'number' then
        local checkpointTimer, checkpointFrequency = torch.Timer(), self.checkpointCondition
        self.checkpointCondition = function()
            local ret = checkpointTimer:time().real > 60*checkpointFrequency
            if ret then checkpointTimer:reset() end
            return ret
        end
    else end


    dlt.log:endSection()
end

T.defaultHooks = {}

function T:run() self:train() end

function T:train()
    dlt.log:section('Training')
    self.data:iterate( { 
        training = function(batch) return self:trainingCallback(batch) end,
        validation = function(batch) return self:validationCallback(batch) end,
        checkpoint = function() self:checkpoint() end
    } )
    dlt.log:endSection()
end


function T:validationCallback(batch)
    self.model:evaluate()
    local prediction = self.model:forward(batch.input)
    self.validationlog:log(self.criterion:forward(prediction,batch.output))
end

-- THIS IS THE DEFAULT MAIN OPTIMIZATION FUNCTION!
-- requires batch = {input = ..., output = ...}
function T:standardCallback(batch)
    self.model:zeroGradParameters()
    local prediction = self.model:forward(batch.input)
    local loss = self.criterion:forward(prediction, batch.output)
    self.trainlog:log(loss)
    local gradOutput = self.criterion:backward(prediction, batch.output)
    self.model.model:backward(batch.input,gradOutput)
    self.optimizer:updateState(self.data.currentEpoch, loss)
    self.optimizer:step( function() return loss,self.model.gradParameters end, self.model.parameters )
end


-- GAN
function T:GANCallback(batch,state)
    self.model.discriminator:training()
    self.model.generator:training()
    local trainGen = self.hooks.trainGenerator()
    if not trainGen then
        -- Discriminator optimization
        self.hooks.onDiscriminatorTrainBegin(self)
        -- Real data
        self.model.discriminator.gradParameters:zero()
        local realDiscriminatorInput = self.hooks.getRealDiscriminatorInput(batch)
        local realPrediction = self.model.discriminator:forward(realDiscriminatorInput)
        local realTarget = self.hooks.getRealDiscriminatorTarget(batch)
        local realDiscriminatorLoss = self.criterion.discriminator:forward(realPrediction,realTarget) 
        local discGradOutput = self.criterion.discriminator:backward(realPrediction,realTarget)
        self.model.discriminator:backward(realDiscriminatorInput,discGradOutput)
        -- Fake data
        local generatorInput = self.hooks.getGeneratorInput(batch)
        local generatorPrediction = self.model.generator:forward(generatorInput)
        local fakeDiscriminatorInput = self.hooks.getFakeDiscriminatorInput(batch,generatorPrediction)
        local fakeDiscriminatorPrediction = self.model.discriminator:forward(fakeDiscriminatorInput)
        local fakeTarget= self.hooks.getFakeDiscriminatorTarget(batch)
        local fakeDiscriminatorLoss = self.criterion.discriminator:forward(fakeDiscriminatorPrediction,fakeTarget) 
        discGradOutput = self.criterion.discriminator:backward(fakeDiscriminatorPrediction,fakeTarget)
        self.model.discriminator:backward(fakeDiscriminatorInput,discGradOutput)
        self.discriminatorLog:log(realDiscriminatorLoss + fakeDiscriminatorLoss)
        self.optimizer.discriminator:step(function() return realDiscriminatorLoss + fakeDiscriminatorLoss,self.model.discriminator.gradParameters end, self.model.discriminator.parameters)
        self.hooks.onDiscriminatorTrainEnd(self)
    else 
        -- Generator optimization
        self.hooks.onGeneratorTrainBegin(self)
        self.model.generator.gradParameters:zero()
        local generatorInput = self.hooks.getGeneratorInput(batch)
        local generatorPrediction = self.model.generator:forward(generatorInput)
        local fakeDiscriminatorInput = self.hooks.getFakeDiscriminatorInput(batch,generatorPrediction)
        local fakeDiscriminatorPrediction = self.model.discriminator:forward(fakeDiscriminatorInput)
        local realTarget = self.hooks.getRealDiscriminatorTarget(batch) -- Invert labels (to minimize positive instead of maximizing negative) 
        local generatorLoss = self.criterion.discriminator:forward(fakeDiscriminatorPrediction,realTarget) 
        local genGradInput = self.criterion.discriminator:backward(fakeDiscriminatorPrediction,realTarget)
        local generatorGradOutput = self.hooks.makeGeneratorGradOutput(self.model.discriminator:updateGradInput(fakeDiscriminatorInput,genGradInput))
        if self.criterion.lambdaG then 
            local generatorTarget = self.hooks.getGeneratorTarget(batch)
            self.criterion.generator:forward(generatorPrediction, generatorTarget)
            -- Do not add loss, just add gradients?
            -- generatorLoss = generatorLoss + self.criterion.generator:forward(generatorPrediction, generatorTarget)
            generatorGradOutput = generatorGradOutput:mul(self.criterion.lambdaD) 
                                + self.criterion.generator:backward(generatorPrediction, generatorTarget):mul(self.criterion.lambdaG)
        end
        
        self.model.generator:backward(generatorInput,generatorGradOutput)
        self.generatorLog:log(generatorLoss)
        self.optimizer.generator:step(function() return generatorLoss, self.model.generator.gradParameters end, self.model.generator.parameters)
    end
    collectgarbage()
end


-- Wasserstein GAN
function T:WGANCallback(batch,state)
    self.model.discriminator:training()
    self.model.generator:training()
    local trainGen = self.hooks.trainGenerator()
    if not trainGen then
        -- Discriminator optimization
        self.hooks.onDiscriminatorTrainBegin(self)
        -- Real data
        self.model.generator.gradParameters:zero()
        self.model.discriminator.gradParameters:zero()
        local realDiscriminatorInput = self.hooks.getRealDiscriminatorInput(batch)
        local realPrediction = self.model.discriminator:forward(realDiscriminatorInput)
        local fakeTarget = self.hooks.getFakeDiscriminatorTarget(batch) -- Use fake target (-1) for correct gradients
        self.model.discriminator:backward(realDiscriminatorInput,fakeTarget) -- - gradient
        local realDiscLoss = realPrediction:mean()
        -- Fake data
        local generatorInput = self.hooks.getGeneratorInput(batch)
        local generatorPrediction = self.model.generator:forward(generatorInput)
        local fakeDiscriminatorInput = self.hooks.getFakeDiscriminatorInput(batch,generatorPrediction)
        local fakeDiscriminatorPrediction = self.model.discriminator:forward(fakeDiscriminatorInput)
        local realTarget = self.hooks.getRealDiscriminatorTarget(batch) -- Use real target (+1) for correct gradients
        self.model.discriminator:backward(fakeDiscriminatorInput,realTarget) -- + gradient
        local fakeDiscLoss = fakeDiscriminatorPrediction:mean()
        local wassersteinDistance =  realDiscLoss - fakeDiscLoss
        self.discriminatorLog:log(wassersteinDistance)
        -- D tries to maximize W-distance, so we minimize -w
        self.optimizer.discriminator:step(function() return -wassersteinDistance,self.model.discriminator.gradParameters end, self.model.discriminator.parameters)
        self.hooks.onDiscriminatorTrainEnd(self)
        -- Generator optimization
    else
        self.hooks.onGeneratorTrainBegin(self)
        self.model.discriminator.gradParameters:zero()
        self.model.generator.gradParameters:zero()
        local generatorInput = self.hooks.getGeneratorInput(batch)
        local generatorPrediction = self.model.generator:forward(generatorInput)
        local fakeDiscriminatorInput = self.hooks.getFakeDiscriminatorInput(batch,generatorPrediction)
        local fakeDiscriminatorPrediction = self.model.discriminator:forward(fakeDiscriminatorInput)
        local fakeTarget = self.hooks.getFakeDiscriminatorTarget(batch)
        local generatorGradOutput = self.hooks.makeGeneratorGradOutput(self.model.discriminator:updateGradInput(fakeDiscriminatorInput,fakeTarget))
        self.model.generator:backward(generatorInput,generatorGradOutput)
        local generatorLoss = -fakeDiscriminatorPrediction:mean()
        self.generatorLog:log(generatorLoss)
        self.optimizer.generator:step(function() return generatorLoss, self.model.generator.gradParameters end, self.model.generator.parameters)
        self.hooks.onGeneratorTrainEnd(self)
    end
    collectgarbage()
end

-- Setup Hooks
-- point is input (z) - sample (x) - output
function T.defaultHooks.GAN(self,hooks)
    self.hooks.getRealDiscriminatorInput = hooks.getRealDiscriminatorInput 
            or function(batch) return batch.sample end 
    self.hooks.getRealDiscriminatorTarget = hooks.getRealDiscriminatorTarget 
            or function(batch) return batch.output:fill(1) end 
    self.hooks.getFakeDiscriminatorTarget = hooks.getFakeDiscriminatorTarget 
            or function(batch) return batch.output:fill(0) end 
    self.hooks.getGeneratorInput = hooks.getGeneratorInput 
            or function(batch) return batch.input end 
    self.hooks.getFakeDiscriminatorInput = hooks.getFakeDiscriminatorInput 
            or function(batch,generatorPrediction) return generatorPrediction end 
    self.hooks.getGeneratorTarget = hooks.getGeneratorTarget 
            or function(batch) return batch.sample end 
    self.hooks.makeGeneratorGradOutput = hooks.makeGeneratorGradOutput 
            or function(discriminatorGradInput) return discriminatorGradInput end 
    self.hooks.onDiscriminatorTrainBegin = hooks.onDiscriminatorTrainBegin
            or function(state) end
    self.hooks.onGeneratorTrainBegin = hooks.onGeneratorTrainBegin
            or function(state) end
    self.hooks.onGeneratorTrainEnd = hooks.onGeneratorTrainEnd
            or function(state) end
    self.hooks.onDiscriminatorTrainEnd = hooks.onDiscriminatorTrainEnd
            or function(state) end
    self.hooks.trainGenerator = hooks.trainGenerator 
            or function()
                    local function gStep() self.nGMany = self.nGMany + 1; self.nDSteps = 0; return true end
                    local function dStep() self.nDSteps = self.nDSteps + 1; return false end
                    if self.nGMany % self.manyFrequency == 0 or self.nGMany < self.manyInitial then 
                        if self.nDSteps == self.nDMany then return gStep() else return dStep() end
                    else if self.nDSteps == self.nDFew then return gStep() else return dStep() end end
            end
end

function T.defaultHooks.WGAN(self,hooks)
    self.hooks.getRealDiscriminatorInput = hooks.getRealDiscriminatorInput 
            or function(batch) return batch.sample end 
    self.hooks.getRealDiscriminatorTarget = hooks.getRealDiscriminatorTarget 
            or function(batch) return batch.output:fill(1) end 
    self.hooks.getFakeDiscriminatorTarget = hooks.getFakeDiscriminatorTarget 
            or function(batch) return batch.output:fill(-1) end 
    self.hooks.getGeneratorInput = hooks.getGeneratorInput 
            or function(batch) return batch.input end 
    self.hooks.getFakeDiscriminatorInput = hooks.getFakeDiscriminatorInput 
            or function(batch,generatorPrediction) return generatorPrediction end 
    self.hooks.getGeneratorTarget = hooks.getGeneratorTarget 
            or function(batch) return batch.sample end 
    self.hooks.makeGeneratorGradOutput = hooks.makeGeneratorGradOutput 
            or function(discriminatorGradInput) return discriminatorGradInput end 
    self.hooks.onDiscriminatorTrainBegin = hooks.onDiscriminatorTrainBegin
            or function(state) end
    self.hooks.onGeneratorTrainBegin = hooks.onGeneratorTrainBegin
            or function(state) end
    self.hooks.onGeneratorTrainEnd = hooks.onGeneratorTrainEnd
            or function(state) end
    self.hooks.onDiscriminatorTrainEnd = hooks.onDiscriminatorTrainEnd
            or function(state)state.model.discriminator.parameters:clamp(self.clampMin,self.clampMax) end
   self.hooks.trainGenerator = hooks.trainGenerator 
            or function()
                    local function gStep() self.nGMany = self.nGMany + 1; self.nDSteps = 0; return true end
                    local function dStep() self.nDSteps = self.nDSteps + 1; return false end
                    if self.nGMany % self.manyFrequency == 0 or self.nGMany < self.manyInitial then 
                        if self.nDSteps == self.nDMany then return gStep() else return dStep() end
                    else if self.nDSteps == self.nDFew then return gStep() else return dStep() end end
            end
end


-- Boundary Equilibrium GAN (BEGAN)
function T:BEGANCallback(batch,state)
    self.model.generator:training()
    self.model.discriminator:training()
    self.model.generator.gradParameters:zero()
    self.model.discriminator.gradParameters:zero()
    self.hooks.onTrainBegin(self)
    
    
    -- Get points
    local generatorInput = self.hooks.getGeneratorInput(batch)
    local realPoint = self.hooks.getDataPoint(batch)
    -- Do generator loss
    local generatorPrediction = self.model.generator:forward(generatorInput)
    local fakePrediction = self.model.discriminator:forward(generatorPrediction)
    local generatorLoss, discGradOutput = self.loss(fakePrediction,generatorPrediction)
    local genGradOutput = self.model.discriminator:backward(generatorPrediction,discGradOutput)
    self.model.generator:backward(generatorInput,genGradOutput)
    
    -- Discriminator loss
    -- fake
    self.model.discriminator.gradParameters:zero()
    generatorInput = self.hooks.getGeneratorInput(batch) -- SHOULD I NOT RESAMPLE Z?
    generatorPrediction = self.model.generator:forward(generatorInput)
    fakePrediction = self.model.discriminator:forward(generatorPrediction)
    fakeLoss, discGradOutput = self.loss(fakePrediction,generatorPrediction)
    self.model.discriminator:backward(generatorPrediction,discGradOutput:mul(-self.ktVar))
    -- real
    local realPrediction = self.model.discriminator:forward(realPoint)
    local realLoss, discGradOutputReal = self.loss(realPrediction,realPoint)
    self.model.discriminator:backward(realPoint,discGradOutputReal)

    local discriminatorLoss = realLoss - self.ktVar*fakeLoss

    self.hooks.onTrainEnd(self)
    -- Log
    self.aeLog:log(realLoss)
    self.discriminatorLog:log(discriminatorLoss)
    self.generatorLog:log(generatorLoss)
    self.ktVarLog:log(self.ktVar)
    local balance = self.diversityRatio*realLoss - generatorLoss
    self.convergenceLog:log(realLoss + torch.abs(balance))

    -- Perform updates
    self.optimizer.discriminator:step(function() return discriminatorLoss ,self.model.discriminator.gradParameters end, self.model.discriminator.parameters)
    self.optimizer.generator:step(function() return generatorLoss, self.model.generator.gradParameters end, self.model.generator.parameters)
    self.ktVar = self.ktVar + self.ktLearningRate*balance
    self.ktVar = math.max(math.min(self.ktVar,1),0)
    -- collectgarbage()
end

function T.defaultHooks.BEGAN(self,hooks)
    self.hooks.getGeneratorInput = hooks.getGeneratorInput 
            or function(batch) return batch.input:uniform(-1,1) end 
    self.hooks.getDataPoint = hooks.getDataPoint 
            or function(batch) return batch.sample end
    self.hooks.onTrainBegin = hooks.onGeneratorTrainBegin
            or function(state) end
    self.hooks.onTrainEnd = hooks.onDiscriminatorTrainEnd
            or function(state) end
end


--- Checkpointing functionality
function T:checkpoint()
    if self.currentEpoch ~= self.data.currentEpoch or self.checkpointCondition(self) then  self:saveCheckpoint()   end
    self.currentEpoch = self.data.currentEpoch
end

function T:makeFilename(name)
    local ret = name;
    if self.saveAll then ret = ret .. string.format(self.chkpFormat,self.chkpCount) end
    return paths.concat(self.savePath,ret .. '.t7')
end

function T:loadCheckpoint()
    self.chkpFile = paths.concat(self.savePath,'checkpoint.t7')
    self.chkpFormat = '%05d'
    if paths.filep(self.chkpFile)  then
        local latest = torch.load(self.chkpFile)
        if latest.currentEpoch >= self.maxEpochs then dlt.log:error('Already did all Epochs!') end
        self.chkpCount = latest.chkpCount + 1
        self.modelFile = latest.model
        self.optimFile = latest.optimizer
        self.currentEpoch = latest.currentEpoch
    else
        self.currentEpoch = 1
        self.chkpCount = 1 
    end
end

function T:saveCheckpoint()
    self.data:syncThreads()
    self.data:syncGPU()
    local checkpoint = {}
    for _,val in ipairs({'model','optimizer'}) do
        if torch.type(self[val]) == 'table' then
            checkpoint[val] = {}
            for objName,obj in pairs(self[val]) do
                local fileName = self:makeFilename(objName .. '_' .. val)
                obj:save(fileName)
                checkpoint[val][objName] = fileName
            end
        else
            local fileName = self:makeFilename(val)
            self[val]:save(fileName)
            checkpoint[val] = fileName
        end
    end
    checkpoint.chkpCount = self.chkpCount
    checkpoint.currentEpoch = self.data.currentEpoch
    torch.save(self.chkpFile,checkpoint)
    dlt.log:yell('Saved checkpoint ' .. self.chkpCount)
    self.chkpCount = self.chkpCount + 1
    collectgarbage()
    collectgarbage()
end