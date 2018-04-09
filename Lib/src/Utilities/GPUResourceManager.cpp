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

/** @file GPUResourceManager.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/GPUResourceManager.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

GPUResourceManager::GPUResourceManager() : Communicator(true){
	numDevices = 0;
	busyDevices = NULL;
#ifndef __APPLE__
	omp_init_lock(&busyDevicesLock);
#endif

	createDeviceTable();
}

GPUResourceManager& GPUResourceManager::getInstance(){
	static GPUResourceManager instance;

	return instance;
}

int GPUResourceManager::allocateDevice(){
	TBTKAssert(
		numDevices > 0,
		"GPUResourceManager::allocateDevice()",
		"No GPU devices available on this machine.",
		"Use CPU version instead."
	);

	int device = 0;
	bool done = false;
	while(!done){
#ifndef __APPLE__
		omp_set_lock(&busyDevicesLock);
#endif
		#pragma omp flush
		{
			for(int n = 0; n < numDevices; n++){
				if(!busyDevices[n]){
					device = n;
					busyDevices[n] = true;
					done = true;
					break;
				}
			}
		}
		#pragma omp flush
#ifndef __APPLE__
		omp_unset_lock(&busyDevicesLock);
#endif
	}

	return device;
}

void GPUResourceManager::freeDevice(int device){
#ifndef __APPLE__
	omp_set_lock(&busyDevicesLock);
#endif
	#pragma omp flush
	{
		busyDevices[device] = false;
	}
	#pragma omp flush
#ifndef __APPLE__
	omp_unset_lock(&busyDevicesLock);
#endif
}

};	//End of namespace TBTK
