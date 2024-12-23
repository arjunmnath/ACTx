shaderSource="$(cat src/Shaders.metal)"
echo "#import <Foundation/Foundation.h>\n
NSString *shaderSource = @R\"(\n$shaderSource\n)\";
" > src/shader.mm

python setup.py build_ext --inplace --verbose
