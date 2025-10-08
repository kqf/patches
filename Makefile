model = model1

all: run.bin $(model).onnx
	echo "Running the script"
	./run.bin $(model).onnx

run.bin: infer.cpp $(model).onnx
	g++ -o run.bin infer.cpp `pkg-config --cflags --libs opencv4` -std=c++11

%.onnx:
	cp ../$@ .

clean:
	rm -rf $(model).onnx

.PRECIOUS: *.onnx
