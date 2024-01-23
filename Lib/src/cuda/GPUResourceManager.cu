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
 *  @author Andreas Theiler	
 */

#include "TBTK/GPUResourceManager.h"
#include "TBTK/Streams.h"
#include <cuda_runtime.h>

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

// Code based on enablePeerAccess from Nvidias CUDA sample library
void GPUResourceManager::enableP2PAccess() {
    
	// Check if any devices are allocated
	if(!busyDevices){
		//TODO error message
		return;
	}
	cudaError_t cudaError = cudaSuccess;
	int currentDevice;
	for(int i = 0; i < numDevices; i++){
		if(busyDevices[i]){
			currentDevice = i;
			// TODO error check cudaSuccess // TODO not sure if this check is needed
			cudaError = cudaGetDevice(&currentDevice);
		}
	}    

    /* Remark: access granted by this cudaDeviceEnablePeerAccess is unidirectional */
    /* Rows and columns represents a connectivity matrix between GPUs in the system */
    for (int activeDevice = 0; activeDevice < numDevices; activeDevice++) {
		if(busyDevices[activeDevice]){
			// TODO error check
			cudaError = cudaSetDevice(activeDevice);
		}
        for (int peer = 0; peer < numDevices; peer++) {
            if (activeDevice != peer && busyDevices[peer]) {
                int canAccessPeer = 0;
				// TODO error check
				cudaError = cudaDeviceCanAccessPeer(&canAccessPeer, activeDevice, peer);
                if (canAccessPeer) {
					// TODO error check
					cudaError = cudaDeviceEnablePeerAccess(peer, 0);
					if(getVerbose()){
						Streams::out << "P2P enabled between device " << 
										activeDevice << " and " <<
										peer <<	endl;
					}
                }
            }
        }
    }
	// TODO error check
	cudaError = cudaSetDevice(currentDevice);
}

};	//End of namespace TBTK
