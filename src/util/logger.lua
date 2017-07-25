local dlt = require('dlt._env')

-- I did not really need a stack, 
-- but sometimes you just feel like writing one.
local stackC = torch.class('Stack')
function stackC:__init()
    self.tab = {}
    self.push = function(sel,...)
                    if ... then 
                        for _,val in ipairs({...}) do 
                            table.insert(sel.tab, val) 
                        end 
                    end
                end
    self.size = function(sel) return #sel.tab end
    self.pop = function(sel,num)
                    num = num or 1
                    num = num < 1 and 1 or num
                    num = num > #sel.tab and #sel.tab or num
                    if num == 0 then return end
                    local ret = {}
                    for i = 1, num do
                        table.insert(ret, sel.tab[#sel.tab])
                        table.remove(sel.tab)
                    end
                    return unpack(ret)
               end
end

-- Logger for dlt
-- Has verbose levels
-- Can be set to write to file
-- Prints things in boxes -- Almost surely unnecessary, but I like it. 
-- Maybe should put a setting to be able to turn boxes off
local L,parent = torch.class('Logger',dlt)
function L:__init(level,filename)
    if filename then self:setFile(filename) end
    level = level or 3
    if level < 1 or level > 6 then
        print('Verbose level needs to be between 1 and 6. Setting to 3')
        level = 3
    end
    self.vlevel = level
    self.levels = {
        error = 1,
        warning = 2,
        yell = 3,
        say = 4,
        detail = 5,
        debug = 6,
        section = 3
    }
    self.width = 78;
    self.sectionStack = Stack()
    return self
end

function L:setLevel(level) self.vlevel = level end
function L:getLevel() return self.vlevel end

function L:print(message,level)
    level = level or 3
    if level == 1 then
        print(message)
        os.exit()
    elseif level <= self.vlevel then
        print(message)
    end
    if self.toFile then
        local file = io.open(self.loggerFileName,'a')
        file:write(message .. '\n')
        file:close()
    end
    return self
end

function L:setFile(filename)
    dlt.help.checkMakeDir(paths.dirname(filename))
    self.loggerFileName = filename
    self.toFile = true
    local exists = paths.filep(self.loggerFileName)
    local file = io.open(self.loggerFileName,'a')
    if exists then
        file:write('\n\nRESTARTING\n\n')
    end
    file:close()
    return self
end

function L:underdash(level,length)
    length = length or self.width
    level = level or self.levels['section']
    self:print(' ' .. string.rep('_',length),level)
    return self
end

function L:paddedText(message,padding,level,length)
    padding = padding or ' '
    length = length or self.width
    level = level or self.levels['section']
    local leftPad = string.rep(padding,torch.floor((length - #message) / 2))
    local rightPad = string.rep(padding,torch.ceil((length - #message) / 2))
    local toPrint = '|' .. leftPad .. message .. rightPad .. '|'
    self:print(toPrint,level)
    return self
end

function L:box(message,level,length)
    level = level or self.levels['section']
    length = length or self.width
    self:underdash(level,length)
    self:paddedText(message,'_',level,length)
    self:padPrint(' ',self.levels['section'])
    return self
end

local function split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then return { str } end

    local result,pat,lastpos = {},"(.-)" .. delim .. "()",nil
    for part, pos in string.gfind(str, pat) do table.insert(result, part); lastPos = pos; end
    table.insert(result, string.sub(str, lastPos))
    return result
end

function L:lineSplit(message,length)
    length = length or self.width
    local words = split(message,' ')
    local lines = {''}
    local count = 1
    for _,val in ipairs(words) do
        if #lines[count] + #tostring(val) + 4 >= length then 
            count = count + 1
            lines[count] = '    '
        end
        lines[count] = lines[count] .. ' ' .. val
    end
    return lines
end

function L:padPrint(message,level)
    local lines = self:lineSplit(message)
    for _,val in ipairs(lines) do
        local padding = self.width - #val - 1
        self:print('| ' .. val .. string.rep(' ',padding) .. '|',level)
    end
    return self
end

function L:error(message)   
    self:print('ERROR: ' .. message .. '\nABORTING',self.levels['error']) 
end
function L:warning(message) 
    self:print('WARNING: ' .. message,self.levels['warning'])  
end
function L:yell(message) 
    self:padPrint(message,self.levels['yell'])  
end
function L:say(message) 
    self:padPrint(message,self.levels['say']) 
end
function L:detail(message) 
    self:padPrint(message,self.levels['detail']) 
end
function L:debug(message) 
    self:print('DEBUG: ' .. message,self.levels['debug']) 
end


function L:section(name)
    self.sectionStack:push(name)
    self:box(name)
    return self
end
function L:endSection() 
    local name = self.sectionStack:pop(1)
    self:padPrint(name .. ' done.',self.levels['section'])
    self:paddedText('','_') 
    return self
end
