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
	static int getNumDevices();

	/** Allocate GPU device. */
	static int allocateDevice();

	/** Free GPU device. */
	static void freeDevice(int device);
private:
	/** Constructor. */
	GPUResourceManager();

	/** Static constructor. */
	static class StaticConstructor{
	public:
		StaticConstructor();
	} staticConstructor;

	/** Number of GPU devices. */
	static int numDevices;

	/** Used to indicate busy devices. */
	static bool *busyDevices;

	/** Lock for busy device table operations. */
	static omp_lock_t busyDevicesLock;

	/** Create device table. */
	static void createDeviceTable();

	/** Destroy device table. */
	static void destroyDeviceTable();
};

inline int GPUResourceManager::getNumDevices(){
	return numDevices;
}

};	//End of namespace TBTK

#endif
