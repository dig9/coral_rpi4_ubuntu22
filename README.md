# Using Google Coral USB on Ubuntu 22.04 / Raspberry pi4

***

 This project is a guide for using Raspberry Pi 4, Ubuntu 22.04 (aarch64) and Google Coral USB. 

 I followed Google's instructions, but I was able to use Coral USB without any major problems only on my PC (amd64/ubuntu 20.04). However, on Raspberry Pi 4's ubuntu 22.04 aarch64, the build failed or did not work normally even after being built. - Including native/cross/docker/colab. On Raspberry Pi 4, all operations were confirmed only on 32bit Debian Buster, and I wasted too much time due to various problems. I hope this page helps users who have been struggling as me.

----------
### 1. Prerequisite

* #### Platform: raspberry pi 4 / Ubuntu 22.04(64bit) installed on NVME/SSD without SD micro
    * Refer to https://www.instructables.com/Raspberry-Pi-4-USB-Boot-No-SD-Card/
* #### Kernel/Package

     * Linux ubuntu 5.15.0-1029-raspi #31-Ubuntu SMP PREEMPT Sat Apr 22 12:26:40 UTC 2023 aarch64 aarch64 aarch64 GNU/Linux
     * Ubuntu 22.04.2 LTS \n \l

* Set bash as default & install minimals.

     ```bash
     sudo dpkg-reconfigure dash
     sudo apt install build-essential vim ssh git tig cmake samba curl
     ```

* Set up **bazelisk**

     * version 1.17.0 : download bazelisk binary at https://github.com/bazelbuild/bazelisk/releases
     * copy it to /usr/local/bin/bazel and apply chmod +x

* Set up **pyenv**

    * install **pyenv**

      ```bash
      git clone htps://github.com/pyenv/pyenv.git ~/.pyenv
      echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
      echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
      echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

      sudo apt install zlib1g-dev bzip2 readline-common sqlite3 openssl libssl-dev xz-utils \
      libbz2-dev libncurses5-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev liblzma-dev python3-tk tk-dev

      . ~/.bashrc
      ```

    * set  virtualenv v3.9.16
      ```bash
      git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
      pyenv install 3.9.16
      pyenv virtualenv 3.9.16 v3.9.16
      ```
      required re-login
      ```bash
      pyenv activate v3.9.16
      pyenv global 3.9.16
      pip install numpy==1.23 (for the time to run application)
      pip install Pillow wheel
      ```


***



## 2. Build tensorflowlite version 2.5.0

* sudo apt install libabsl-dev libusb-1.0-0-dev


   ``` bash
    mkdir ~/github && cd github
    git clone git@github.com:tensorflow/tensorflow.git tf250
    cd tf250 && git check out v2.5.0 -b v2.5.0
    ./configure
   	## configure : check the python location whether indicate pyenv environments.
   ```

* Build tensorflowlite
    ```bash
    bazel build -c opt //tensorflow/lite:tensorflowlite
	```

    You may have a build  error ***${bazel cache}/external/ruy/ruy/block_map.cc***, therefore, you shall add next statement into the file.
    ```cpp
    #include <limits>
    ```

    When you used upper version than 2.5.0 and  the error occurred related with bazel version,
    you need to declare ```export USE_BAZEL_VERSION=5.4.0 ``` . But, it could produce another issues regarding edgetpu.




* Installing libflatbuffers which used for build tensorflowlite onto system library

   ```bash
   cd ~
   mkdir ~/tf250_lib ~/tf250_external && \
   cp ~/github/tf250/bazel-tf250/bazel-out/aarch64-opt/bin/tensorflow/lite/libtensorflowlite.so ~/tf250_lib/
   cp -rfl ~/github/tf250/bazel-tf250/external/* ~/tf250_external/
   ```



* Go to the flatbuffers folder and build with cmake:

    ``` bash
    cd ~/tf250_external/flatbuffers && mkdir build_cmake && cd build_cmake
    cmake -DCMAKE_CXX_FLAGS="-Wno-error=class-memaccess" ..
    make -j4
    sudo  make install
    ## You shall check a file /usr/local/lib/libflatbuffers.a, and  find a version:1.12.0 in /usr/local/include/flatbuffers/base.h
    ```




***

### 3. libedgetpu.so : for cpp & python

##### 	Cautions: Never try to build aarch64 libedgetpu.so for raspberry pi4. Native/CrossCompile/Docker/CoLab, any build ways from websites doesn't provide proper & exact solution you may need. Even though built the library successfully, there were so many fxxx terrible issues on Ubuntu 22.04 64bit.

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt update
sudo apt install libedgetpu1-std
```





***

### 4. python module : tflite_runtime : only for python

#### 	Build in tflite_runtime reference source code.

```
 cd ~/github/tf250/tensorflow/lite/tools/pip_package
```

Patch build_pip_package_with_bazel.sh with following statement

``` bash
     @@ -48,6 +48,8 @@ cp -r "${TENSORFLOW_LITE_DIR}/tools/pip_package/debian" \
            "${TENSORFLOW_LITE_DIR}/python/interpreter_wrapper" \
            "${BUILD_DIR}"
      cp "${TENSORFLOW_LITE_DIR}/python/interpreter.py" \
     +   "${TENSORFLOW_LITE_DIR}/python/metrics_interface.py" \
     +   "${TENSORFLOW_LITE_DIR}/python/metrics_portable.py" \
         "${BUILD_DIR}/tflite_runtime"
```

Build pip packages
``` bash
./build_pip_packages_with_bazel.sh
pip install --force-reinstall gen/tflite_pip/python3/dist/tflite_runtime-2.5.0-cp39-cp39-linux_aarch64.whl
```



The content below is not required. For reference only

Misc. 

```
 cd tensorflow/lite/tools/make
 ./download_dependecies.sh && ./build_aarch64_lib.sh
 ##it could be errors in ruy/block_map.cc and absl/CMakeLists.txt

 ## 1. the errors would occurr in ruy/ruy/block_map.cc regarding for <limits>
 ##    it needs to add these headers into the source file
 ##     #include <stdexcept>
 ##     #include <limits>
 ## 2. CMakeList.txt
 ## 	 it needs to link system library libraries with renamed.
```





***

### 5. Running example / cpp







***

### 6. Running example / python

***



### 6. Trouble shooting



| No   | Trouble Cases                                                | Remark |
| :--- | ------------------------------------------------------------ | ------ |
| 1    | RuntimeError: <br />module compiled against API version 0x10 but this version of numpy is 0xd<br />ImportError: numpy.core.multiarray failed to import | Python |
|      | check the pynum version ```pip lists numpy```. Probably, you need to re-install pynum:1.23<br />```pip install numpy==1.23``` |        |
| 2    | ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.<br/>tflite-runtime 2.5.0 requires numpy~=1.19.2, but you have numpy 1.23.0 which is incompatible. | Python |
|      | You can ignore this error statement.                         |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |

