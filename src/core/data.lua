local dlt = require('dlt._env')

local D,parent = torch.class('dlt.Data',dlt)

-- dlt.Data objects utilize loaders to iterate over datasets 
-- using single or multiple threads
-- Requires a loader that implements init(), assignBatch(), mode(), size(), 
--                          [reshuffle()] (with or without arguments)
function D:__init(loader,pointSize, datasets, currentEpoch)
    dlt.parse(self)
    dlt.configure(self)
    if loader == nil then 
        dlt.log:error('No loader provided for data.') 
    end
    if pointSize == nil then 
        dlt.log:error('No pointSize provided for data.') 
    end
    self.datasets = datasets or {'training'}
    self.currentEpoch = currentEpoch or 1
    if self.currentEpoch > self.maxEpochs then 
        dlt.log:error('Max epochs exeeded (' .. self.maxEpochs .. ').') 
    end

    dlt.log:section(('Data initialization'))
    -- Launch Donkeys in threads or on master thread
    if self.nThreads > 0 then
        dlt.log:yell(string.format('Initializing %d thread(s)',self.nThreads))
        -- local threads = require('threads')
        threads.Threads.serialization('threads.sharedserialize')
        local mid = threads.Mutex():id()
        self.datathreads = threads.Threads(
            self.nThreads, function() 
                dl = require('dlt')
                torch.setdefaulttensortype('torch.FloatTensor') 
            end,
            function(idx)
                tid = idx
                torch.manualSeed(self.seed)
                dlt = dl
                local t = require('threads')
                _donkey = dlt.Donkey(loader, pointSize, self.batchSize,
                                     self.useLocks, self.collectGarbage,
                                     self.tensorType)
                _donkey.loader:mode('training')
                mutex = t.Mutex(mid)
            end
        );
    else
        _donkey = dlt.Donkey(loader,pointSize,self.batchSize,self.useLocks, 
                                self.collectGarbage,self.tensorType)
        _donkey.loader:mode('training')
        self.datathreads = {}
        function self.datathreads:addjob(f1, f2)  f2(f1())  end
        function self.datathreads:synchronize() end
    end
    -- Get nPoints and nBatches for datasets
    for _,datasetName in ipairs(self.datasets) do
        self[datasetName] = {}
        self.datathreads:addjob(
            function() 
                return _donkey.loader:size(datasetName) 
            end,
            function(nPoints) 
                self[datasetName].nPoints = nPoints 
                self[datasetName].nBatches = 
                        math.ceil(self[datasetName].nPoints / self.batchSize)
            end )
    end
    self:syncThreads()
    -- Create batch of master thread (or gpu memory)
    local device = self.nGPU > 0 and 'gpu' or 'cpu'
    self.batch = dlt.help.createBatch(self.batchSize, pointSize, 
                                        self.tensorType, device)

    -- Create Timers
    self.iterationTimer = torch.Timer()
    self.epochTimer = torch.Timer()
    self.transferTimer = torch.Timer()
    self.computeTimer = torch.Timer()
    -- Initializations
    self.currentSetID = 1
    self.currentPoint = {}  
    for _,set in ipairs(self.datasets) do 
        self.currentPoint[set] = 1 
    end

    -- Report dataset sizes
    for _,set in ipairs(self.datasets) do
        dlt.log:yell( string.gsub(set,'^%l', string.upper) .. ' dataset: ' 
                        .. self[set].nPoints .. ' points, ' 
                        .. self[set].nBatches .. ' batches.')
    end
    dlt.log:endSection()
end

function D:iterate(callbacks)
    self.callbacks = callbacks
    self:syncGPU()
    self.iterationTimer:reset()
    repeat self:next() until self.terminate
    self:syncGPU()
    self:syncThreads()
    dlt.log:yell(self.terminateMessage)
end



function D:next()
    local currentSet = self:currentSet()
    local iPoint = self.currentPoint[currentSet]
    self:addjob(function() return _donkey:getBatch(iPoint,currentSet) end,
                function(donkeyBatch,donkeyTime)
                    
                    self.transferTimer:reset()
                    
                    self:syncGPU()
                    self:transfer(donkeyBatch)

                    local trt = self.transferTimer:time().real
                    self.computeTimer:reset()
                    self.terminate, self.terminateMessage = 
                            self.callbacks[self:currentSet()](self.batch)
                    
                    self:syncGPU()
                    local computeTime = self.computeTimer:time().real
                    local iterTime = self.iterationTimer:time().real                    
                    self.iterationTimer:reset()
                    dlt.log:detail(string.format( 
                            'load: %.3fs, iteration: %.3fs,' .. 
                            ' transfer: %.3fs, compute: %.3fs.',
                            donkeyTime, iterTime, trt, computeTime
                        ))
                end)
    self:nextPoint()
    if self.callbacks.checkpoint then self.callbacks.checkpoint() end
end

-- Increase counter and raise flags (for epoch change, checkpointing)
function D:nextPoint()
    local curP = self.currentPoint[self:currentSet()] + self.batchSize
    self.currentPoint[self:currentSet()] = curP
                                            
    if curP > self[self:currentSet()].nPoints then
        curP = (curP - 1) % self[self:currentSet()].nPoints + 1
        self.currentPoint[self:currentSet()] = curP
        self:nextSet()
        if self.currentSetID == 1 then  self:nextEpoch() end
    end
end

function D:nextSet()
    self:syncThreads()
    self.currentSetID = self.currentSetID % #self.datasets + 1
    local currentSet = self:currentSet()
    if self.epochReshuffle then 
        self:runOnAllThreads(function() 
                                _donkey.loader:reshuffle() 
                             end) 
    end
    self:runOnAllThreads(function() 
                                _donkey.loader:mode(currentSet) 
                         end)
end

function D:currentSet() 
    return self.datasets[self.currentSetID] 
end

function D:nextEpoch()
    self:syncThreads()
    dlt.log:yell(string.format('Epoch %d took %.3fs to complete.',
                        self.currentEpoch, self.epochTimer:time().real))
    self.epochTimer:reset()
    self.currentEpoch = self.currentEpoch + 1
    -- if epochs are over then signal termination
    if self.currentEpoch > self.maxEpochs then
        self.terminate = true
        self.terminateMessage = 'Done with all epochs!'
    end
end

function D:transfer(donkeyBatch)
    if self.nGPU > 0 then dlt.help.copyPoint(donkeyBatch,self.batch)
        else self.batch = donkeyBatch end    
end

function D:runOnAllThreads(fun,callback) 
    if self.nThreads > 0 then
        self.datathreads:specific(true)
        for i=1,self.nThreads do self.datathreads:addjob(i,fun) end
        self.datathreads:specific(false)
    else 
        callback = callback or function() end
        self.datathreads:addjob(fun,callback) 
    end
end

function D:addjob(fun,callback) 
    self.datathreads:addjob(fun,callback) 
end
function D:syncThreads() 
    self.datathreads:synchronize() 
end
function D:syncGPU() 
    if self.nGPU > 0 then 
        cutorch.synchronizeAll() 
    end
end
function D:getEpoch() 
    return self.currentEpoch 
end