/** @package TBTKcalc
 *  @file GPUResourceManager.h
 *  @brief GPU resource manager.
 *
 *  @ author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GPU_RESOURCE_MANAGER
#define COM_DAFER45_TBTK_GPU_RESOURCE_MANAGER

#include <omp.h>

namespace TBTK{

class GPUResourceManager{
public:
	/** Get number of GPU devices. */
	int getNumDevices();

	/** Allocate GPU device. */
	int allocateDevice();

	/** Free GPU device. */
	void freeDevice(int device);

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

	/** Lock for busy device table operations. */
	omp_lock_t busyDevicesLock;

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
