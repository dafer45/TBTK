/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file GPUResourceManager.cu
 *  
 *  @author Kristofer Björnson
 */

#include "TBTK/GPUResourceManager.h"
#include "TBTK/Streams.h"

using namespace std;

namespace TBTK{

void GPUResourceManager::createDeviceTable(){
	cudaGetDeviceCount(&numDevices);

	if(getGlobalVerbose())
		Streams::out << "Num GPU devices: " << numDevices << "\n";

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
