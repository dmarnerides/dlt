-- Colorspace conversions
-- Assumptions:
-- RGB is CIE RGB
-- RGB is float and 
-- image is depth * width * height (e.g. 3*1920*1080 for FHD)
-- Images are all colored (3 channels)
-- For each colorspace, indices correspond to the letters 
--  (e.g. im[1] = R, im[2] = G, im[3] = B)
local dlt = require('dlt._env')

dlt.color = {}
local C = dlt.color

local rt2, rt3, rt6 = math.sqrt(2), math.sqrt(3), math.sqrt(6)
local irt2, irt3, irt6 = 1/rt2, 1/rt3, 1/rt6
local epsilon = 1e-15

C.mat = {

    rgb2xyz           = torch.Tensor({{ 0.4887180,  0.3106803,  0.2006017 },
                                      { 0.1762044,  0.8129847,  0.0108109 },
                                      { 0.0000000,  0.0102048,  0.9897952 }}),

    xyz2rgb           = torch.Tensor({{ 2.3706743, -0.9000405, -0.4706338 },
                                      {-0.5138850,  1.4253036,  0.0885814 },
                                      { 0.0052982, -0.0146949,  1.0093968 }}),
               
    xyz2lms           = torch.Tensor({{ 0.3897100,  0.6889800, -0.0786800 },
                                      {-0.2298100,  1.1834000,  0.0464100 },
                                      { 0.0000000,  0.0000000,  1.0000000 }}),

    lms2xyz           = torch.Tensor({{ 1.9102000, -1.1121200,  0.2019080 },
                                      { 0.3709500,  0.6290540,  0.0000000 },
                                      { 0.0000000,  0.0000000,  1.0000000 }}),
    -- CIE XYZ to LMS D65
    xyz2lmsD65        = torch.Tensor({{ 0.4002000,  0.7075000, -0.0807000 },
                                      {-0.2280000,  1.1500000,  0.0612000 },
                                      { 0.0000000,  0.0000000,  0.9184000 }}), 
    -- L'M'S' TO IPT
    lpmpsp2ipt        = torch.Tensor({{ 0.4000000,  0.4000000,  0.2000000 },
                                      { 4.4550000, -4.8510000,  0.3960000 },
                                      { 0.8056000,  0.3572000, -1.1628000 }}), 
    -- IPT to L'M'S'
    ipt2lpmpsp        = torch.Tensor({{ 1.0000000,  0.0975689,  0.2052260 },
                                      { 1.0000000, -0.1138760,  0.1332170 },
                                      { 1.0000000,  0.0326151, -0.6768870 }}), 
    -- LMS D65 to CIE XYZ
    lmsD652xyz        = torch.Tensor({{ 1.8502400, -1.1383000,  0.2384350 }, 
                                      { 0.3668310,  0.6438850, -0.0106734 },
                                      { 0.0000000,  0.0000000,  1.0888500 }}),

    loglms2lalphabeta = torch.Tensor({{   irt3   ,    irt3   ,    irt3    },
                                      {   irt6   ,    irt6   , -2*irt6    },
                                      {   irt2   ,   -irt2   ,     0      }}),

    lalphabeta2loglms = torch.Tensor({{   irt3   ,    irt6   ,    irt2    },
                                      {   irt3   ,    irt6   ,   -irt2    },
                                      {   irt3   , -2*irt6   ,     0      }})
}

-- There must be a better way to do this
-- Multiplies each pixel of input (3,w,h) with matrix mat
function C.matrixMultiply(input,mat)
    local output = input.new():resizeAs(input):zero()
    output[1]:add(mat[1][1],input[1])
             :add(mat[1][2],input[2])
             :add(mat[1][3],input[3])
    output[2]:add(mat[2][1],input[1])
             :add(mat[2][2],input[2])
             :add(mat[2][3],input[3])
    output[3]:add(mat[3][1],input[1])
             :add(mat[3][2],input[2])
             :add(mat[3][3],input[3])
    return output
end

-- CIE RGB - CIE XYZ
function C.rgb2xyz(im) 
    return C.matrixMultiply(im,C.mat.rgb2xyz) 
end
-- CIE XYZ - CIE RGB
function C.xyz2rgb(im) 
    return C.matrixMultiply(im,C.mat.xyz2rgb) 
end
-- CIE XYZ - LMS (equal energy)
function C.xyz2lms(im) 
    return C.matrixMultiply(im,C.mat.xyz2lms) 
end
-- LMS (equal energy) - CIE XYZ
function C.lms2xyz(im) 
    return C.matrixMultiply(im,C.mat.lms2xyz) 
end
-- CIE RGB - LMS (equal energy)
function C.rgb2lms(im) 
    return C.xyz2lms(C.rgb2xyz(im)) 
end
-- LMS (equal energy) - CIE RGB
function C.lms2rgb(im) 
    return C.xyz2rgb(C.lms2xyz(im)) 
end
-- LMS - Lαβ
function C.lms2lalphabeta(im) 
    return C.matrixMultiply(torch.log(im+epsilon),C.mat.loglms2lalphabeta) 
end
-- Lαβ - LMS
function C.lalphabeta2lms(im) 
    return torch.exp(C.matrixMultiply(im,C.mat.lalphabeta2loglms)) 
end
-- CIE RGB - Lαβ
function C.rgb2lalphabeta(im) 
    return C.lms2lalphabeta(C.rgb2lms(im)) 
end
-- Lαβ - CIE RGB
function C.lalphabeta2rgb(im) 
    return C.lms2rgb(C.lalphabeta2lms(im)) 
end
-- CIE XYZ - LMS D65
function C.xyz2lmsD65(im) 
    return C.matrixMultiply(im,C.mat.xyz2lmsD65) 
end
-- LMS D65 - CIE XYZ 
function C.lmsD652xyz(im) 
    return C.matrixMultiply(im,C.mat.lmsD652xyz) 
end
-- L'M'S' - IPT
function C.lpmpsp2ipt(im) 
    return C.matrixMultiply(im,C.mat.lpmpsp2ipt) 
end
-- IPT - L'M'S'
function C.ipt2lpmpsp(im) 
    return C.matrixMultiply(im,C.mat.ipt2lpmpsp) 
end
-- LMS D65 - L'M'S'
function C.lmsD652lpmpsp(im) 
    local res = torch.abs(im:clone())
    res:pow(0.43)
    res:cmul(torch.sign(im))
    return res
end
-- L'M'S' - LMS D65
function C.lpmpsp2lmsD65(im) 
    local res = torch.abs(im:clone())
    res:pow(1/0.43)
    res:cmul(torch.sign(im))
    return res
end

-- CIE XYZ - IPT
function C.xyz2ipt(im) 
    return C.lpmpsp2ipt(C.lmsD652lpmpsp(C.xyz2lmsD65(im))) 
end
-- IPT - CIE XYZ
function C.ipt2xyz(im) 
    return C.lmsD652xyz(C.lpmpsp2lmsD65(C.ipt2lpmpsp(im))) 
end
-- CIE RGB - IPT
function C.rgb2ipt(im) 
    return C.xyz2ipt(C.rgb2xyz(im)) 
end
-- IPT - CIE RGB
function C.ipt2rgb(im) 
    return C.xyz2rgb(C.ipt2xyz(im)) 
end
-- CIE RGB - LMS D65
function C.rgb2lmsD65(im) 
    return C.xyz2lmsD65(C.rgb2xyz(im)) 
end
-- LMS D65 - CIE RGB
function C.lmsD652rgb(im) 
    return C.xyz2rgb(C.lmsD652xyz(im)) 
end
-- CIE RGB - L'M'S'
function C.rgb2lpmpsp(im) 
    return C.lmsD652lpmpsp(C.rgb2lmsD65(im)) 
end
-- L'M'S' - CIE RGB
function C.lpmpsp2rgb(im) 
    return C.lmsD652rgb(C.lpmpsp2lmsD65(im)) 
end

-- MISC
function C.linearizeSRGB_(img)
    if not C.linearLookup then 
        C.linearLookup = torch.FloatTensor(256)
        count = 0
        C.linearLookup:apply( function()
                        local x = count/255
                        count = count + 1
                        if x <= 0.04045 then
                            return x/12.92
                        else
                            return torch.pow((x + 0.055) / (1.055),2.4)
                        end
                    end )
    end
    return img:apply(function(x)
                        return C.linearLookup[x*255 + 1]
                    end )
end


return C