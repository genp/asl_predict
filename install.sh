mkdir lstm
cd lstm
git clone https://github.com/jeffdonahue/caffe.git
# adjust makefile
cd caffe
make all
make test
make runtest
make pycaffe
