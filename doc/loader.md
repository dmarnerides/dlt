# Loader
Loaders are to be used with [`dlt.Data`](data.md) but can also be used on their own.

Implemented loaders: *MNIST*, *CIFAR*, *CelebA*, *Places*, *pix2pix*.

It's straightforward to create a loader for a new dataset that's compatible with the rest of the toolbox.

## Usage
```lua
-- LoaderName is a placeholder name here.
data = dlt.LoaderName( s )
```
s is a table of settings. Some commonly used settings are:

* `path` Path to data.
* `shuffle` Whether to shuffle the indices when loading.
* `assignPoint` Function that assigns a loaded datapoint into a given batch.

Some loaders may accept additional settings, e.g:

* `transform` Function that is applied on the dataset (for datasets that are loaded on initialization).
* `name` String for loaders that handle more than one dataset (e.g. CIFAR may be `10` or `100`).
* `type` For datasets that use image.load at every call to :get().
* `download` For datasets that are easily downloaded (*MNIST*, *CIFAR*)
* `fileList` For *Places* datasets. Files that contain custom lists (e.g. the list for the colored images only). e.g.
    * `fileList = {training = '/path/to/col_places.csv'} `

## Directory structures
### MNIST
 Path should contain *train_32x32.t7* and *test_32x32.t7* from extracting [this](https://s3.amazonaws.com/torch7/data/mnist.t7.tgz).
 Can use `download` setting to automatically get the data (provided path must already exist).
### CIFAR
 Path should contain *cifar10-train.t7*, *cifar100-train.t7*, *cifar10-test.t7*, *cifar100-test.t7* created using [cifar.torch](https://github.com/soumith/cifar.torch).
 Can use `download` setting to automatically get the data  (provided path must already exist).
### CelebA
Path must contain all the original images.
### Places (Places205)
Path must contain directories a,b,c... as well as train_places205.csv and val_places205.csv (unless custom lists are provided through `fileList`)
### Places2 (Places365)
Path must contain the (renamed) directories *training*, *validation*, *testing*. Each of these directory contents are unchanged from the extracted (standard 256x256), only renamed. The three directories must also contain *places365_val.txt*, *places365_val.txt* and *places365_test.txt* respectively (unless custom lists are provided through `fileList`). The *.txt* lists are found from extracting `filelist_places365-standard.tar`.
### pix2pix
Path must contain the directories *cityscapes*, *maps*, *facades*, *edges2handbags*, *edges2shoes* as downloaded from [here](https://github.com/phillipi/pix2pix).

## Methods

### `init([setName])`
Initializes `setName` (*training, validation, testing*). If not provided, all available sets are initialized.

**NOTE**: This MUST be called right after the creation of the loader, before any other functions are used. It is separated from the `__init()` method so that it can be called independently on multiple threads (e.g. to initialize non serializable objects).

### `mode(setName)`
Changes set (*training, validation, testing*).

### `get(index)`
Returns datapoint at `index` from current set. If shuffle is on, then it returns the shuffled index.

### `[s] size([setName])`
Returns the size of `setName` (or current set if not provided).

### `reshuffle()`
Reshuffles!

### `assignBatch(batch,iDataPoint,n)`
Fills given batch with n consecutive points starting from `iDataPoint` according to the `assignPoint` function (provided on initialization).

## Example 1

```lua
local places = dlt.Places{ path = '~/data/places365', shuffle = true, type = 'byte'}

-- Must call init
-- Initializes all (training,validation,testing for places)
-- current mode is training by default
places:init() 

-- get returns image and class for Places2
local img, cls = places:get(1)
image.display(img)
print(cls)

-- Reshuffling
places:reshuffle()
img, cls = places:get(1)  -- should be different
image.display(img)
print(cls)

-- Get validation
places:mode('validation')
img, cls = places:get(1) -- should be from validation
image.display(img)
print(cls)
```

(Run with qlua)

## Example 2
```lua
local dlt = require('dlt')
local places = dlt.Places{ path = '~/data/places365', 
                           shuffle = true, 
                           type = 'byte',
                           -- assignPoint describes the rule that gets a loaded image and class and puts it into a batch
                           assignPoint = function(batch,iBatchMember,img,cls)
                                if img:size(1) == 1 then img:repeatTensor(3,1,1) end -- Greyscale images
                                img = image.scale(img,64,64)
                                img = img:float():div(255) -- This should be avoided by just using type = 'float'
                                batch.discriminatorInput[iBatchMember]:copy(img)
                                batch.generatorInput[1][iBatchMember]:copy(img:mul(3):clamp(0,1))
                                batch.generatorInput[2][iBatchMember] = cls
                            end }

places:init('training') -- Initialize only training 
local batchSize = 4
local myBatch = {discriminatorInput = torch.Tensor(batchSize,3,64,64), 
                generatorInput = { torch.Tensor(batchSize,3,64,64), torch.Tensor(batchSize) } }

places:assignBatch(myBatch,100,batchSize)
image.display(myBatch.discriminatorInput)
image.display(myBatch.generatorInput[1])
```

(Run with qlua)