#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      cout << msg << endl;
  }
} gLogger;

void cvImageToTensor(const cv::Mat & image, float *tensor, nvinfer1::Dims dimensions)
{
  const size_t channels = dimensions.d[0];
  const size_t height = dimensions.d[1];
  const size_t width = dimensions.d[2];
  const size_t stridesCv[3] = { width * channels, channels, 1 };
  const size_t strides[3] = { height * width, width, 1 };

  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      for (int k = 0; k < channels; k++) 
      {
        const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = (float) image.data[offsetCv];
      }
    }
  }
}

size_t numTensorElements(nvinfer1::Dims dimensions)
{
  if (dimensions.nbDims == 0)
    return 0;
  size_t size = 1;
  for (int i = 0; i < dimensions.nbDims; i++)
    size *= dimensions.d[i];
  return size;
}

float argmax(float *tensor, nvinfer1::Dims dimensions)
{ 
  size_t max_ind=0;
  size_t i=0;
  size_t numel=numTensorElements(dimensions);
  for(; i<numel; i++)
  {	cout<<i<<endl;
	cout<<*(tensor+i)<<endl;
	if( (*(tensor+i)) > (*(tensor+max_ind)) ) max_ind=i ;
  }
  return max_ind;
}


int main()
{

string imageFilename = "../dataset/S_test.jpg";
string planFilename="../TensorRT_Inference/Engine/engine.plan";
string inputnodeName="x";
string outputnodeName="logits/BiasAdd";
string classes_names="../classes_names.txt";
//getting the classes names
vector<string> classes;
ifstream ReadFile;
ReadFile.open(classes_names);
string str;
if (ReadFile.is_open())
{
	while(!ReadFile.eof())
	{	getline(ReadFile,str);
		classes.push_back(str);
	}
}
classes.pop_back();
for(int i=0; i<classes.size(); i++)
{	
	cout<<i<<endl;
	cout<<classes[i]<<endl;
}

//Load the engine 
cout<<"Loading The TensorRT engine from plan file"<<endl;
ifstream planFile(planFilename);
if(!planFile.is_open()) {cout<<"Could not open plan file."<<endl; return 1;}

stringstream planBuffer;
planBuffer << planFile.rdbuf();
string plan=planBuffer.str();

//Create a runtime object to deserialize inference engine
IRuntime* runtime=createInferRuntime(gLogger);
ICudaEngine* engine= runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);

// Create space to store intermediate activation values
IExecutionContext *context = engine->createExecutionContext();

//Get the input / output dimensions 
int inputBindingIndex, outputBindingIndex;
inputBindingIndex = engine->getBindingIndex(inputnodeName.c_str());
outputBindingIndex = engine->getBindingIndex(outputnodeName.c_str());
if(inputBindingIndex < 0) {cout << "Invalid input name." << endl; return 1;}
if(outputBindingIndex < 0) {cout << "invalid output name." << endl; return 1;}

Dims inputDims, outputDims;
inputDims = engine->getBindingDimensions(inputBindingIndex);
outputDims = engine->getBindingDimensions(outputBindingIndex);
int inputWidth, inputHeight;
inputHeight = inputDims.d[1];
inputWidth = inputDims.d[2];


//Read image convert color and resize
cout << "Preprocessing input ..." << endl;
cv::Mat image = cv::imread(imageFilename,CV_LOAD_IMAGE_COLOR);
cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );

if(image.data == NULL ) { cout << "Could not read image from file." << endl; return 1;}
//cv::cvtColor(image, image, cv::COLOR_RGB2BGR, 3); 
cv::resize(image, image, cv::Size(inputWidth, inputHeight));//, cv::INTER_CUBIC);
//image.convertTo(image, CV_32FC3);
cv::imshow("Display window", image);


//Convert from uint8+NHWC to float+NCHW
float *inputDataHost, *outputDataHost;
size_t numInput, numOutput;
numInput = numTensorElements(inputDims);
numOutput = numTensorElements(outputDims);
inputDataHost = (float*) malloc(numInput * sizeof(float));
outputDataHost = (float*) malloc(numOutput * sizeof(float));
cvImageToTensor(image, inputDataHost, inputDims);

//Transfer to device
float *inputDataDevice, *outputDataDevice;
cudaMalloc(&inputDataDevice, numInput * sizeof(float));
cudaMalloc(&outputDataDevice, numOutput * sizeof(float));
cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
void *bindings[2];
bindings[inputBindingIndex] = (void*) inputDataDevice;
bindings[outputBindingIndex] = (void*) outputDataDevice;


//Execute engine
cout << "Executing inference engine ..." << endl;
const int kBatchSize = 1;
context->execute(kBatchSize, bindings);

//Transfer output back to host
cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);

/* parse output */
//  vector<size_t> sortedIndices = argsort(outputDataHost, outputDims);
//cout << "\nThe top-5 indices are: ";
//  for (int i = 0; i < 5; i++)
//    cout << sortedIndices[i] << " ";

//Read Output
cout<<"The prediction is :" << classes[argmax(outputDataHost,outputDims)] << endl; 
//clean up
cv::waitKey(0);
runtime->destroy();
engine->destroy();
context->destroy();
free(inputDataHost);
free(outputDataHost);
cudaFree(inputDataDevice);
cudaFree(outputDataDevice);

return 0;
}
