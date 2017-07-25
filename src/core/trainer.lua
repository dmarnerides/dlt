local dlt = require('dlt._env')

local T,parent = torch.class('dlt.Trainer',dlt)

function T:__init(experiment)
    dlt.parse(self)
    dlt.configure(self)
    dlt.reportExperiment(self)
    dlt.log:section('Trainer initialization')

    -- Generic settings
    self.useGPU = self.nGPU > 0
    self.savePath = self.savePath
    -- format
    self.format = self.nGPU > 0 and 'gpu' or 'cpu'
    self.format = 'torch.' .. dlt.help.tensorList[self.format][self.tensorType]
 
    -- Configure training type
    local trainingTypes = {'simple', 'validate', 'GAN', 
                            'WGAN','BEGAN','custom'}
    self.trainingType = experiment.trainingType or 'simple'
    if not dlt.help.inTable(trainingTypes,self.trainingType) then 
        dlt.log:error('Unkown training type: ' .. self.trainingType) 
    end
    self.datasets = self.trainingType == 'validate' 
                        and {'training','validation'} 
                         or {'training'}
    
    -- Load checkpoint
    self:loadCheckpoint()

    -- Create data
    self.data = dlt.Data(experiment.loader,experiment.pointSize,
                                        self.datasets,self.currentEpoch)

    -- Setup training
    if self.trainingType == 'simple' or self.trainingType == 'validate' then
        self.trainingCallback = self.standardCallback 
        local modelCreate = self.modelFile or experiment.model.create
        self.model = dlt.Model(modelCreate,experiment.model.name,
                                                    experiment.model.save)
        self.optimizer = dlt.Optimizer(experiment.optim,self.tensorType,
                                                self.useGPU,self.optimFile)
        self.criterion = experiment.criterion
        self.criterion:type(self.format)
        self.log = {
            training = dlt.Trainlog('training', self.savePath)
        }
        if self.trainingType == 'validate' then 
            self.log.validation = dlt.Trainlog('validation', self.savePath)
        end
    elseif dlt.help.inTable({'GAN','WGAN'},self.trainingType) then
        self.model, self.optimizer = {},{}
        self.trainingCallback = self[self.trainingType .. 'Callback']

        if self.trainingType == 'WGAN' then
            local clamp = experiment.clamp or {}
            self.clampMin = clamp[1] or -0.01
            self.clampMax = clamp[2] or 0.01
            -- Training frequencies
            self.nDSteps, self.nGSteps = 0, 0
            local steps = experiment.steps or {}
            local defSteps = {5,100,25,100}
            for i,val in ipairs(defSteps) do steps[i] = steps[i] or val end
            self.trainGenerator = function()
                local ng = self.nGSteps
                local many = ((ng % steps[4] == 0) or (ng < steps[3]))
                local size = many and steps[2] or steps[1]
                if self.nDSteps == size - 1 then 
                    self.nGSteps = self.nGSteps + 1 
                    self.nDSteps = 0 
                    return true 
                else 
                    self.nDSteps = self.nDSteps + 1
                    return false 
                end
            end

        end 
        -- Create models,optimizers and criterions
        self.modelFile = self.modelFile or {}
        self.optimFile = self.optimFile or {}
        self.criterion = experiment.criterion
        for _,val in ipairs{'discriminator','generator'} do
            local modelCreate = self.modelFile[val] 
                                    or experiment.model[val].create
            local modelName = experiment.model[val].name or val
            self.model[val] = dlt.Model(modelCreate,modelName,
                                                experiment.model[val].save)
            self.optimizer[val] = dlt.Optimizer(experiment.optim[val],
                                                self.tensorType,self.useGPU,
                                                self.optimFile[val])
            if self.criterion and self.criterion[val] then 
                self.criterion[val]:type(self.format) 
            end
        end
        self.log = {
            discriminator = dlt.Trainlog('discriminator', self.savePath),
            generator = dlt.Trainlog('generator', self.savePath)
        }
    elseif self.trainingType == 'BEGAN' then
        self.model, self.optimizer = {},{}
        self.trainingCallback = self.BEGANCallback
        self.modelFile = self.modelFile or {}
        self.optimFile = self.optimFile or {}
        for _,val in ipairs{'discriminator','generator'} do
            local modelCreate = self.modelFile[val] 
                                    or experiment.model[val].create
            local modelName = experiment.model[val].name or val
            self.model[val] = dlt.Model(modelCreate,modelName,
                                            experiment.model[val].save)
            self.optimizer[val] = dlt.Optimizer(experiment.optim[val],
                                                self.tensorType,self.useGPU,
                                                        self.optimFile[val])
        end

        self.diversityRatio = experiment.diversityRatio or 0.5
        self.ktVar = experiment.ktVarInit or 0
        self.ktLearningRate = experiment.ktLearningRate or 0.001
        self.loss = experiment.loss or nn.AbsCriterion()
        self.loss:type(self.format)

        self.log = {}
        for _,val in ipairs{'discriminator','generator','autoencoder',
                            'kt','convergence'} do
            self.log[val] = dlt.Trainlog(val,self.savePath)
        end
    elseif self.trainingType == 'custom' then
        -- Replace default callbacks if given 
        if experiment.trainingCallback == nil then 
            dlt.log:error('Trainer custom mode requires training callback.')
        end
        self.trainingCallback = experiment.trainingCallback
        -- Model
        if experiment.model.create then
            local modelCreate = self.modelFile or experiment.model.create
            self.model = dlt.Model(modelCreate,experiment.model.name,
                                                    experiment.model.save)
        else -- multiple
            self.model = {}
            self.modelFile = self.modelFile or {}
            for name,val in pairs(experiment.model) do
                local modelCreate = self.modelFile[name] or val.create
                local modelName = val.name or name
                self.model[name] = dlt.Model(modelCreate,modelName,val.save)
            end
        end
        -- Criterion
        if torch.type(experiment.criterion) ~= 'table' then
            self.criterion = experiment.criterion
            self.criterion:type(self.format)
        else -- multiple
            self.criterion = {}
            for name,val in pairs(experiment.criterion) do
                self.criterion[name] = val
                self.criterion[name]:type(self.format)
            end
        end
        -- Optimizer
        if not experiment.optim or experiment.optim.name then
            self.optimizer = dlt.Optimizer(experiment.optim,self.tensorType,
                                                    self.useGPU,self.optimFile)
        else -- multiple
             self.optimizer = {}
            self.optimFile = self.optimFile or {}
            for name,val in pairs(experiment.optim) do
                self.optimizer[name] = dlt.Optimizer(val,self.tensorType,
                                                        self.useGPU,
                                                        self.optimFile[name])
            end
        end
        -- Logs
        if experiment.log then
            self.log = {}
            for _,name in pairs(experiment.log) do
                self.log[name] = dlt.Trainlog(name,self.savePath)
            end
        end

    end

    -- Replace default callbacks if given 
    if experiment.trainingCallback then
        dlt.log:yell('Replacing default training callback with given.')
        self.trainingCallback = experiment.trainingCallback
    end

    -- Checkpoint condition
    self.checkpointCondition = experiment.checkpointCondition 
                                    or function() return false end 
    
    if torch.type(self.checkpointCondition) == 'number' 
                and self.checkpointCondition <=0 then
        dlt.log:error('Checkpoint Condition must be positive number' .. 
                                                ' (minutes) or function')
    end
    if torch.type(self.checkpointCondition) ~= 'function' 
                and torch.type(self.checkpointCondition) ~= 'number' then 
            dlt.log:error('Checkpoint Condition must be positive number' .. 
                                                ' (minutes) or function')
    end
    if torch.type(self.checkpointCondition) == 'number' then
        local checkpointTimer = torch.Timer()
        local checkpointFrequency = self.checkpointCondition
        self.checkpointCondition = function()
            local ret = checkpointTimer:time().real > 60*checkpointFrequency
            if ret then checkpointTimer:reset() end
            return ret
        end
    else end

    dlt.log:endSection()
end

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
    self.log.validation:log(self.criterion:forward(prediction,batch.output))
end

-- THIS IS THE DEFAULT MAIN OPTIMIZATION FUNCTION!
-- requires batch = {input = ..., output = ...}
function T:standardCallback(batch)
    self.model:zeroGradParameters()
    local prediction = self.model:forward(batch.input)
    local loss = self.criterion:forward(prediction, batch.output)
    self.log.training:log(loss)
    local gradOutput = self.criterion:backward(prediction, batch.output)
    self.model.model:backward(batch.input,gradOutput)
    self.optimizer:updateState(self.data.currentEpoch, loss)
    self.optimizer:step( function() 
                            return loss,self.model.gradParameters 
                        end, 
                        self.model.parameters )
end


-- GAN
-- Assumes batch is 'input' --> G --> 'sample' --> D --> 'output'
function T:GANCallback(batch)
    -- Some shortcuts
    local D = self.model.discriminator
    local critD = self.criterion.discriminator
    local optD = self.optimizer.discriminator
    local G = self.model.generator
    local optG = self.optimizer.generator

    D:training()
    G:training()

    -- Discriminator optimization
    -- Real data
    D.gradParameters:zero()
    local rPred = D:forward(batch.sample)
    local rDLoss, dGradOut = critD(realPrediction, batch.output:fill(1)) 
    D:backward(batch.sample,dGradOut)
    
    -- Fake data
    local gPred = G:forward(batch.input:uniform(-1,1))
    local fPred = D:forward(gPred)
    local fDLoss, dGradOut = critD(fPred, batch.output:fill(0)) 
    D:backward(gPred,dGradOut)

    local dLoss = rDLoss + fDLoss
    self.log.discriminator:log(dLoss)
    optD:step(function() return dLoss, D.gradParameters end, D.parameters)

    -- Generator optimization
    G.gradParameters:zero()
    -- Invert labels (to minimize positive instead of maximizing negative) 
    local gLoss, dGradOut = critD(fPred, batch.output:fill(1)) 
    local gGradOut = D:updateGradInput(gPred,dGradOut)
    G:backward(batch.input,gGradOut)
    self.log.generator:log(gLoss)
    optG:step(function() return gLoss, G.gradParameters end, G.parameters)

end


-- Wasserstein GAN
-- Assumes batch is 'input' --> G --> 'sample' --> D --> 'output'
function T:WGANCallback(batch)
        -- Some shortcuts
        local D = self.model.discriminator
        local optD = self.optimizer.discriminator
        local G = self.model.generator
        local optG = self.optimizer.generator
        local batchSize = batch.sample:size(1)
        -- Make sure we are in training mode
        D:training()
        G:training()

        -- Discriminator optimization
        -- Real data
        D.gradParameters:zero()
        local rPred = D:forward(batch.sample)
        D:backward(batch.sample,batch.output:fill(1))
        local rDLoss = rPred:mean()
        
        -- Fake data
        local gPred = G:forward(batch.input:normal(0,1))
        local fPred = D:forward(gPred)
        local gGradOut = D:backward(gPred,batch.output:fill(-1))
        local fDLoss = fPred:mean()

        local wDistance =  rDLoss - fDLoss
        self.log.discriminator:log(wDistance)
        D.gradParameters:mul(-1/batchSize)
        optD:step(function() return -wDistance, D.gradParameters end, D.parameters)
        D.parameters:clamp(self.clampMin,self.clampMax)
        -- Generator optimization
        if self.trainGenerator() then 
            G.gradParameters:zero()
            G:backward(batch.input,gGradOut)
            local gLoss = fDLoss
            self.log.generator:log(gLoss)
            G.gradParameters:div(batchSize)
            optG:step(function() return gLoss, G.gradParameters end, G.parameters)
        end
end

-- Boundary Equilibrium GAN (BEGAN)
function T:BEGANCallback(batch)
    -- Some shortcuts
    local D = self.model.discriminator
    local optD = self.optimizer.discriminator
    local G = self.model.generator
    local optG = self.optimizer.generator

    G:training()
    D:training()
    
    D.gradParameters:zero()
    self.dGradParamBuffer = self.dGradParamBuffer or D.gradParameters:clone()
    self.dGradParamBuffer:zero()
    -- Discriminator loss
    -- real
    local rPred = D:forward(batch.sample)
    local rLoss, dGradOut = self.loss(rPred,batch.sample)
    D:backward(batch.sample,dGradOut)
    self.dGradParamBuffer:add(D.gradParameters)
    -- fake
    D.gradParameters:zero()
    G.gradParameters:zero()
    local gPred = G:forward(batch.input:uniform(-1,1))
    local fPred = D:forward(gPred)
    local fLoss, dGradOut = self.loss(fPred,gPred)
    local gGradOut = D:backward(gPred,dGradOut)
    self.dGradParamBuffer:add(D.gradParameters:mul(-self.ktVar))
    
    local dLoss = rLoss - self.ktVar*fLoss

    -- Generator loss
    -- SHOULD I RESAMPLE Z?
    local gLoss = fLoss
    G:backward(batch.input,gGradOut)
    
    -- Log
    self.log.autoencoder:log(rLoss)
    self.log.discriminator:log(dLoss)
    self.log.generator:log(gLoss)
    self.log.kt:log(self.ktVar)
    local balance = self.diversityRatio*rLoss - gLoss
    self.log.convergence:log(rLoss + torch.abs(balance))

    -- Perform updates
    optD:step(function() return dLoss ,self.dGradParamBuffer end, D.parameters)
    optG:step(function() return gLoss, G.gradParameters end, G.parameters)
    self.ktVar = self.ktVar + self.ktLearningRate*balance
    self.ktVar = math.max(math.min(self.ktVar,1),0)
end

--- Checkpointing functionality
function T:checkpoint()
    if self.currentEpoch ~= self.data.currentEpoch 
                    or self.checkpointCondition(self) then  
        self:saveCheckpoint()   
    end
    self.currentEpoch = self.data.currentEpoch
end

function T:makeFilename(name)
    local ret = name;
    if self.saveAll then 
        ret = ret .. string.format(self.chkpFormat,self.chkpCount) 
    end
    return paths.concat(self.savePath,ret .. '.t7')
end

function T:loadCheckpoint()
    self.chkpFile = paths.concat(self.savePath,'checkpoint.t7')
    self.chkpFormat = '%05d'
    if paths.filep(self.chkpFile)  then
        local latest = torch.load(self.chkpFile)
        if latest.currentEpoch >= self.maxEpochs then 
            dlt.log:error('Already did all Epochs!') 
        end
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