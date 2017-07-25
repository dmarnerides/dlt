local dlt = require('dlt._env')
dlt.plot = {}
local P = dlt.plot

local inputFile = dlt.help.checkHomePath('~/results/itmpixel2/save/training.log')

-- Loads csv file and returns an array of tables each containing the key/name
-- of the column and a 1D FloatTensor with the data.
function P.loadToTensors(file)
    local loaded = csvigo.load({path = file , verbose = false})
    local ret = {}
    for key,val in pairs(loaded) do
        ret[#ret + 1] = { data = torch.FloatTensor(dlt.help.apply(val,tonumber)),
                           key = key}
    end
    return ret
end

function P.saveToFile(t,fileName)
    fileName = fileName or paths.tmpname()
    local f = io.open(fileName,'w')
    for i=1,t:size(1) do  
        f:write(string.format('%1.4e\n',t[i])) 
    end
    f:close()
    return fileName
end

-- Creates a simple 
function P.createEPS(filename,callback,epsName,title,ylabel,xlabel)
    epsName = epsName or 'plot.eps'
    callback = callback or function(t) return t[1].data end
    local outFile = P.saveToFile(callback(P.loadToTensors(filename)))
    local command = string.format( [[ 
    set terminal postscript eps enhanced size 10in,7in
    set nokey 
    set output "%s"
    set title "%s"
    set ylabel "%s"
    set xlabel "%s"
    plot "%s" using 1 w l
    ]],epsName, title, ylabel, xlabel, outFile)

    gnuplot.raw( command)
    return epsName
end

P.func = {}
F = P.func

-- e.g. funcs = {{f1,f1paramtable}, {f2,f2paramtable} }
function F.compose(funcs)
    return function(t)
                for i,f in ipairs(funcs) do
                    if type(f[2]) ~= 'table' then 
                        t = f[1](t,f[2])
                    else
                        t = f[1](t,unpack(f[2]))  
                    end 
                end
                return t
           end

end

function F.getColumn(t,i) 
    return t[i].data 
end
function F.movingAvg(t,size)
    local size = size or 10
    local ret = torch.FloatTensor(t:size(1) - size + 1)
    local count = 0
    ret:apply(function()
                count = count + 1
                return t[{{count, count + size - 1}}]:mean()
              end )
    return ret
end
function F.avg(t,size)
    local size = size or 10
    local ret = torch.FloatTensor(math.floor(t:size(1)/size))
    local count = 0
    ret:apply(function()
                count = count + 1
                return t[{{(count-1)*size + 1, count*size}}]:mean()
              end )
    return ret
end
function F.movingVar(t,size)
    local size = size or 10
    local ret = torch.FloatTensor(t:size(1) - size + 1)
    local count = 0
    ret:apply(function()
                count = count + 1
                return t[{{count, count + size - 1}}]:var()
              end )
    return ret
end
function F.var(t,size)
    local size = size or 10
    local ret = torch.FloatTensor(math.floor(t:size(1)/size))
    local count = 0
    ret:apply(function()
                count = count + 1
                return t[{{(count-1)*size + 1, count*size}}]:var()
              end )
    return ret
end

function F.getTail(t,n)
    local length = t:size(1)
    return t[{{length - n + 1,length}}]
end
function F.removeTail(t,n)
    local length = t:size(1)
    return t[{{1,length - n}}]
end
function F.getHead(t,n)
    return t[{{1,n}}]
end
function F.removeHead(t,n)
    local length = t:size(1)
    return t[{{1+n,length}}]
end