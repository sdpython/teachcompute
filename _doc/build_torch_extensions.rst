Build Torch Extensions
======================

Torch allows the user to define its own Torch Extensions with
this mechanism : :epkg:`Custom C++ and CUDA Extensions`.
It does not work very well without CUDA so a different build path
is taken in cmake definition.

cmake
+++++

Everything is defined in `torch_extensions.cmake
<https://github.com/sdpython/teachcompute/blob/main/_cmake/targets/torch_extensions.cmake>`_.
It is a work going on.

setup.py
++++++++

`setup.py <https://github.com/sdpython/teachcompute/blob/main/setup.py>`_
defines a custom command to call cmake.
A line was added to copy the binaries into the main folder.