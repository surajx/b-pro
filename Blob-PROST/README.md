# Howto get it running

* Fork ALE from https://github.com/mcmachado/Arcade-Learning-Environment
* Fork B_PRO from https://github.com/mcmachado/b-pro

#### Configuring ALE
* makefile.unix
    * SET: USE_SDL=1
    * CMD: make all -f makefile.unix -j8

#### Configuring B-PRO
* b-pro/Blob-PROST/mainBlobTime.cpp
    * CHANGE
        * -ale.setFloat("frame_skip", param.getNumStepsPerAction());
        * +ale.setInt("frame_skip", param.getNumStepsPerAction());

* b-pro/conf/bpro.cfg
    * SET: DISPLAY = 1

* SYSTEM
    *SET: LD_LIBRARY_PATH="$LD_LIBRARY_PATH:<path to ale>"

* Makefile
    * SET: USE_SDL := 1
    *CMD: make all
