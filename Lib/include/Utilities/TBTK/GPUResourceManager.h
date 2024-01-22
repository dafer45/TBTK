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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file GPUResourceManager.h
 *  @brief GPU resource manager.
 *
 *  @ author Kristofer Björnson
 *  @author Andreas Theiler	
 */

#ifndef COM_DAFER45_TBTK_GPU_RESOURCE_MANAGER
#define COM_DAFER45_TBTK_GPU_RESOURCE_MANAGER

#include "Communicator.h"

#ifndef __APPLE__
#	include <omp.h>
#endif

namespace TBTK{

class GPUResourceManager : public Communicator{
public:
	/** Get number of GPU devices. */
	int getNumDevices();

	/** Allocate GPU device. */
	int allocateDevice();

	/** Free GPU device. */
	void freeDevice(int device);

	/** enables peer to peer communication if available	*/
	void enableP2PAccess();

	/** Get singleton instance. */
	static GPUResourceManager& getInstance();
private:
	/** Constructor. */
	GPUResourceManager();

	/** Delete copy constructor. */
	GPUResourceManager(const GPUResourceManager &gpuResourceManager) = delete;

	/** Delete operator=. */
	GPUResourceManager& operator=(const GPUResourceManager &gpuResourceManager) = delete;

	/** Number of GPU devices. */
	int numDevices;

	/** Used to indicate busy devices. */
	bool *busyDevices;

#ifndef __APPLE__
	/** Lock for busy device table operations. */
	omp_lock_t busyDevicesLock;
#endif

	/** Create device table. */
	void createDeviceTable();

	/** Destroy device table. */
	void destroyDeviceTable();
};

inline int GPUResourceManager::getNumDevices(){
	return numDevices;
}

};	//End of namespace TBTK

#endif
/// @endcond
