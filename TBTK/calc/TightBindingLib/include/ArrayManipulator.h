/** @package TBTKcalc
 *  @file ArrayManipulator.h
 *  @brief Provides methods for manipulation of arrays.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY_MANIPULATOR
#define COM_DAFER45_TBTK_ARRAY_MANIPULATOR

#include "Index.h"

namespace TBTK{
namespace Util{

/** A static class providing methods for manipulation of arrays. */
template<typename T>
class ArrayManipulator{
public:
	static void* create(const Index &ranges);
	static void* create(const Index &ranges, T fill);
	static void destroy(void *array, const Index &ranges);
	static T* flatten(void *array, const Index &ranges);
	static void* unflatten(T *array, const Index &ranges);
private:
	static void* createRecursive(Index ranges, void **result);
	static void* createRecursive(Index ranges, void **result, T fill);
	static void destroyRecursive(void *array, Index ranges);
	static void flattenRecursive(void *array, Index ranges, T *result, int offset);
	static void unflattenRecursive(T *array, Index ranges, void **result, int offset);
};

template<typename T>
void* ArrayManipulator<T>::create(const Index &ranges){
	void *result;

	createRecursive(ranges, &result);

	return result;
}

template<typename T>
void* ArrayManipulator<T>::createRecursive(Index ranges, void **result){
	if(ranges.indices.size() == 1){
		*((T**)result) = new T[ranges.indices.at(0)];
	}
	else{
		*((void**)result) = new void*[ranges.indices.at(0)];

		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			createRecursive(ranges, &(((void**)(*result))[n]));
	}
}

template<typename T>
void* ArrayManipulator<T>::create(const Index &ranges, T fill){
	void *result;

	createRecursive(ranges, &result, fill);

	return result;
}

template<typename T>
void* ArrayManipulator<T>::createRecursive(Index ranges, void **result, T fill){
	if(ranges.indices.size() == 1){
		*((T**)result) = new T[ranges.indices.at(0)];
		for(int n = 0; n < ranges.indices.at(0); n++)
			(*((T**)result))[n] = fill;
	}
	else{
		*((void**)result) = new void*[ranges.indices.at(0)];

		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			createRecursive(ranges, &(((void**)(*result))[n]), fill);
	}
}

template<typename T>
void ArrayManipulator<T>::destroy(void *array, const Index &ranges){
	destroyRecursive(array, ranges);
}

template<typename T>
void ArrayManipulator<T>::destroyRecursive(void *array, Index ranges){
	if(ranges.indices.size() == 1){
		delete [] (T*)array;
	}
	else{
		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			destroyRecursive(((void**)array)[n], ranges);

		delete [] (void**)array;
	}
}

template<typename T>
T* ArrayManipulator<T>::flatten(void *array, const Index &ranges){
	int size = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++)
		size *= ranges.indices.at(n);

	T *result = new T[size];

	flattenRecursive(array, ranges, result, 0);

	return result;
}

template<typename T>
void ArrayManipulator<T>::flattenRecursive(void *array, Index ranges, T *result, int offset){
	if(ranges.indices.size() == 1){
		for(int n = 0; n < ranges.indices.at(0); n++){
			result[offset + n] = ((T*)array)[n];
		}
	}
	else{
		int offsetMultiplier = 1;
		for(int n = 1; n < ranges.indices.size(); n++)
			offsetMultiplier *= ranges.indices.at(n);

		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			flattenRecursive(((void**)array)[n], ranges, result, offset + offsetMultiplier*n);
	}
}

template<typename T>
void* ArrayManipulator<T>::unflatten(T *array, const Index &ranges){
	void *result;

	unflattenRecursive(array, ranges, &result, 0);

	return (void*)result;
}

template<typename T>
void ArrayManipulator<T>::unflattenRecursive(T *array, Index ranges, void **result, int offset){
	if(ranges.indices.size() == 1){
		*((T**)result) = new T[ranges.indices.at(0)];
		for(int n = 0; n < ranges.indices.at(0); n++)
			(*((T**)result))[n] = array[offset + n];
	}
	else{
		*((void**)result) = new void*[ranges.indices.at(0)];

		int offsetMultiplier = 1;
		for(int n = 1; n < ranges.indices.size(); n++)
			offsetMultiplier *= ranges.indices.at(n);

		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			unflattenRecursive(array, ranges, &(((void**)(*result))[n]), offset + offsetMultiplier*n);
	}
}

};	//End of namespace Util
};	//End of namespace TBTK

#endif

