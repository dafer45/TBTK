/** @file GPUResourceManager.cu
 *  
 *  @author Kristofer Björnson
 */

#include "../include/GPUResourceManager.h"
#include "../include/Streams.h"

using namespace std;

namespace TBTK{

void GPUResourceManager::createDeviceTable(){
	cudaGetDeviceCount(&numDevices);

	Util::Streams::out << "Num GPU devices: " << numDevices << "\n";

	if(numDevices > 0){
		busyDevices = new bool[numDevices];
		for(int n = 0; n < numDevices; n++)
			busyDevices[n] = false;
	}
}

void GPUResourceManager::destroyDeviceTable(){
	if(numDevices > 0){
		delete [] busyDevices;
		busyDevices = NULL;
	}
}

};	//End of namespace TBTK
