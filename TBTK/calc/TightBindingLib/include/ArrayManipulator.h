/** @package TBTKcalc
 *  @file ArrayManipulator.h
 *  @brief Provides methods for manipulation of arrays.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY_MANIPULATOR
#define COM_DAFER45_TBTK_ARRAY_MANIPULATOR

#include "Index.h"
#include <iostream>

namespace TBTK{
namespace Util{

/** A static class providing methods for manipulation of arrays. */
template<typename T>
class ArrayManipulator{
public:
	/** Allocate uninitialized array. */
	static void* create(const Index &ranges);

	/** Allocate initialized array. */
	static void* create(const Index &ranges, T fill);

	/** Deallocate array. */
	static void destroy(void *array, const Index &ranges);

	/** Flatten multi-dimensional array. */
	static T* flatten(void *array, const Index &ranges);

	/** Create multi-dimensional array from one-dimensional array. */
	static void* unflatten(T *array, const Index &ranges);

	/** Print array. */
	static void print(void * array, const Index &ranges);
private:
	/** Recursive helper function for uninitialized create. */
	static void* createRecursive(Index ranges, void **result);

	/** Recursive helper function for initialized create. */
	static void* createRecursive(Index ranges, void **result, T fill);

	/** Recursive helper function for destroy. */
	static void destroyRecursive(void *array, Index ranges);

	/** Recursive helper function for flatten. */
	static void flattenRecursive(void *array, Index ranges, T *result, int offset);

	/** Recursive helper function for unflatten. */
	static void unflattenRecursive(T *array, Index ranges, void **result, int offset);

	/** Recursive helper function for print. */
	static void printRecursive(void *array, Index ranges);
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

template<typename T>
void ArrayManipulator<T>::print(void *array, const Index &ranges){
	printRecursive(array, ranges);
}

template<typename T>
void ArrayManipulator<T>::printRecursive(void *array, Index ranges){
	if(ranges.indices.size() == 1){
		for(int n = 0; n < ranges.indices.at(0); n++)
			std::cout << ((T*)array)[n] << "\t";
		std::cout << "\n";
	}
	else{
		int currentRange = ranges.indices.at(0);
		ranges.indices.erase(ranges.indices.begin());
		for(int n = 0; n < currentRange; n++)
			printRecursive(((void**)array)[n], ranges);
		std::cout << "\n";
	}
}

};	//End of namespace Util
};	//End of namespace TBTK

#endif
