rm *.onnx
rm *.json
rm *.png
rm *.csv
rm *.nsys-rep
rm *.sqlite
rm tt_*
rm plot*
rm test* -rf
rm temp* -rf
rm dump* -rf
rm *.sarif
rm *.svg
rm dump_models -rf
rm neural_coder_workspace -rf

rm teachcompute/validation/cython/*_cy.cpp
rm teachcompute/validation/cython/vector_function_cy.cpp
rm teachcompute/validation/cython/dot_cython.cpp
rm teachcompute/validation/cython/dot_cython_omp.cpp
rm teachcompute/validation/cython/experiment_cython.cpp
rm teachcompute/validation/cython/mul_cython_omp.cpp
rm teachcompute/validation/cython/td_mul_cython.cpp
rm teachcompute/validation/cython/dot_blas_lapack.cpp

rm _doc/examples/plot*.onnx
rm _doc/examples/plot*.txt
rm _doc/examples/ort*.onnx
rm _doc/examples/*.sarif
rm _doc/examples/*.json
rm _doc/examples/*.png
rm _doc/examples/*.csv
rm _doc/examples/*.xlsx
rm _doc/examples/dummy*.onnx
rm _doc/examples/*.opt.onnx
rm _doc/examples/*.dynamo.onnx
rm _doc/examples/*.script.onnx
rm _doc/examples/dump_models -rf