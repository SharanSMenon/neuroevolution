mkdir build/
cd build
# Make sure to replace /opt/homebrew/Cellar/libtorch/ with your libtorch path.
# Also replace the compiler path with your path.
cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/Cellar/libtorch/ -DCMAKE_C_COMPILER=/usr/bin/gcc ..
cmake --build . --config Release
./simple-evolution