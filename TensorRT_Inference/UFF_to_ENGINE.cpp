#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <NvInfer.h>
#include <NvUffParser.h>

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

class Logger : public ILogger
{
	void log(Severity severity, const char * msg) override
	{
		cout << msg << endl;
	}
} gLogger;


int main()
{
	
	string uffFilename ="../TensorRT_Inference/UFF.uff";
	string planFilename ="../TensorRT_Inference/Engine/engine.plan";
	string inputName = "x";
	int inputHeight = 100;
	int inputWidth = 100;
	string outputName = "logits/BiasAdd";
	int maxBatchSize = 1;
	int maxWorkspaceSize= 0; 
	DataType dataType=DataType::kFLOAT;

	//parse uff
	IBuilder *builder = createInferBuilder(gLogger);
	INetworkDefinition *network = builder->createNetwork();
	IUffParser *parser = createUffParser();
	parser->registerInput(inputName.c_str(), DimsCHW(3, inputHeight, inputWidth),UffInputOrder::kNCHW);
	parser->registerOutput(outputName.c_str());
	 if (!parser->parse(uffFilename.c_str(), *network, dataType))
	  {
	    cout << "Failed to parse UFF\n";
	    builder->destroy();
	    parser->destroy();
	    network->destroy();
	    return 1;
	  }
	

	// build engine 
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(maxWorkspaceSize);
	ICudaEngine *engine = builder->buildCudaEngine(*network);

	// serialize engine and write to file
	ofstream planFile;
	planFile.open(planFilename);
	IHostMemory *serializedEngine = engine->serialize();
	planFile.write((char *)serializedEngine->data(), serializedEngine->size());
	planFile.close();
	
	builder->destroy();
	parser->destroy();
	network->destroy();
	engine->destroy();
	serializedEngine->destroy();
	
	return 0;
}
