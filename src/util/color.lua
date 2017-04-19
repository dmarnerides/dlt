-- Colorspace conversions
-- Assumptions:
-- RGB is CIE RGB
-- RGB is float and 
-- image is depth * width * height (e.g. 3*1920*1080 for FHD)
-- Images are all color (3 channels)
-- For each colorspace, indices correspond to the letters (e.g. im[1] = R, im[2] = G, im[3] = B)
local dlt = require('dlt._env')

dlt.color = {}
local C = dlt.color

local rt2, rt3, rt6 = math.sqrt(2), math.sqrt(3), math.sqrt(6)
local irt2, irt3, irt6 = 1/rt2, 1/rt3, 1/rt6
local epsilon = 1e-15

local normalizeGlobal = function(im) im:add(-im:min()) return im:mul(im:max()) end
local normalizeChannel = function(im) for i=1,3 do im[i]:add(-im[i]:min()) im[i]:mul(im[i]:max()) end return im end
--{{},{},{}}

C.mat = {
    rgb2xyz = {{0.4887180,  0.3106803,  0.2006017},{0.1762044,  0.8129847,  0.0108109},{0.0000000,  0.0102048,  0.9897952}},
    xyz2rgb = {{2.3706743, -0.9000405, -0.4706338},{-0.5138850,  1.4253036,  0.0885814},{0.0052982, -0.0146949,  1.0093968}},
    xyz2lms = {{0.38971, 0.68898, -0.07868},{-0.22981, 1.18340, 0.04641},{0,0,1}},
    lms2xyz = {{1.9102,-1.11212,0.201908},{0.37095,0.629054,0},{0,0,1}},
    xyz2lmsD65 = {{0.4002, 0.7075, -0.0807},{-0.2280, 1.1500, 0.0612},{0,0,0.9184}}, -- CIE XYZ to LMS D65
    lpmpsp2ipt = {{0.4000,0.4000,0.2000},{4.4550,-4.8510,0.3960},{0.8056,0.3572,-1.1628}}, -- L'M'S' TO IPT
    ipt2lpmpsp = {{1, 0.0975689, 0.205226}, {1, -0.113876, 0.133217}, {1, 0.0326151, -0.676887}}, -- IPT to L'M'S'
    lmsD652xyz =  {{1.85024, -1.1383, 0.238435}, {0.366831, 0.643885, -0.0106734}, {0, 0, 1.08885}}, -- LMS D65 to CIE XYZ
    loglms2lalphabeta = {{irt3,irt3,irt3},{irt6,irt6,-2*irt6},{irt2,-irt2,0}},
    lalphabeta2loglms = {{irt3,irt6,irt2},{irt3,irt6,-irt2},{irt3,-2*irt6,0}}
}


function C.matrixMultiply(input,mat)
    local output = input.new():resizeAs(input):zero()
    output[1]:add(mat[1][1],input[1]):add(mat[1][2],input[2]):add(mat[1][3],input[3])
    output[2]:add(mat[2][1],input[1]):add(mat[2][2],input[2]):add(mat[2][3],input[3])
    output[3]:add(mat[3][1],input[1]):add(mat[3][2],input[2]):add(mat[3][3],input[3])
    return output
end

-- CIE RGB - CIE XYZ
function C.rgb2xyz(im) return C.matrixMultiply(im,C.mat.rgb2xyz) end
function C.xyz2rgb(im) return C.matrixMultiply(im,C.mat.xyz2rgb) end
-- CIE XYZ - LMS (equal energy)
function C.xyz2lms(im) return C.matrixMultiply(im,C.mat.xyz2lms) end
function C.lms2xyz(im) return C.matrixMultiply(im,C.mat.lms2xyz) end
-- CIE RGB - LMS (equal energy)
function C.rgb2lms(im) return C.xyz2lms(C.rgb2xyz(im)) end
function C.lms2rgb(im) return C.xyz2rgb(C.lms2xyz(im)) end
-- LMS - Lαβ
function C.lms2lalphabeta(im) return C.matrixMultiply(torch.log(im+epsilon),C.mat.loglms2lalphabeta) end
function C.lalphabeta2lms(im) return torch.exp(C.matrixMultiply(im,C.mat.lalphabeta2loglms)) end
-- CIE RGB - Lαβ
function C.rgb2lalphabeta(im) return C.lms2lalphabeta(C.rgb2lms(im)) end
function C.lalphabeta2rgb(im) return C.lms2rgb(C.lalphabeta2lms(im)) end

-- CIE XYZ - LMS D65
function C.xyz2lmsD65(im) return C.matrixMultiply(im,C.mat.xyz2lmsD65) end
function C.lmsD652xyz(im) return C.matrixMultiply(im,C.mat.lmsD652xyz) end
-- L'M'S' - IPT
function C.lpmpsp2ipt(im) return C.matrixMultiply(im,C.mat.lpmpsp2ipt) end
function C.ipt2lpmpsp(im) return C.matrixMultiply(im,C.mat.ipt2lpmpsp) end

-- LMS D65 - L'M'S'
function C.lmsD652lpmpsp(im) 
    local res = torch.abs(im:clone())
    res:pow(0.43)
    res:cmul(torch.sign(im))
    return res
end
function C.lpmpsp2lmsD65(im) 
    local res = torch.abs(im:clone())
    res:pow(1/0.43)
    res:cmul(torch.sign(im))
    return res
end

-- CIE XYZ - IPT
function C.xyz2ipt(im) return C.lpmpsp2ipt(C.lmsD652lpmpsp(C.xyz2lmsD65(im))) end
function C.ipt2xyz(im) return C.lmsD652xyz(C.lpmpsp2lmsD65(C.ipt2lpmpsp(im))) end

-- CIE RGB - IPT
function C.rgb2ipt(im) return C.xyz2ipt(C.rgb2xyz(im)) end
function C.ipt2rgb(im) return C.xyz2rgb(C.ipt2xyz(im)) end

-- CIE RGB - LMS D65
function C.rgb2lmsD65(im) return C.xyz2lmsD65(C.rgb2xyz(im)) end
function C.lmsD652rgb(im) return C.xyz2rgb(C.lmsD652xyz(im)) end

-- CIE RGB - L'M'S'
function C.rgb2lpmpsp(im) return C.lmsD652lpmpsp(C.rgb2lmsD65(im)) end
function C.lpmpsp2rgb(im) return C.lmsD652rgb(C.lpmpsp2lmsD65(im)) end

return C