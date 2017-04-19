package = "dlt"
version = "1.0-1"

source = {
   url = "https://github.org/dmarnerides/dlt.git",
   tag = "master"
}

description = {
    summary = "Deep Learning Toolbox for Torch",
    detailed = [[
        This package provides a set of tools to easily create/run/replicate 
        deep learning experiments using Torch.
   ]],
   homepage = "https://github.com/dmarnerides/dlt",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "paths",
   "class",
   "optim",
   "nn",
   "threads"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}