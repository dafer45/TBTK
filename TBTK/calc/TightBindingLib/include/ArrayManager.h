/** @package TBTKcalc
 *  @file ArrayManiager.h
 *  @brief Provides methods for manipulation of arrays.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY_MANAGER
#define COM_DAFER45_TBTK_ARRAY_MANAGER

#include "Index.h"
#include "Streams.h"

namespace TBTK{
namespace Util{

/** A static class providing methods for manipulation of arrays. The class is
 *  intended to provide methods that simplifies the creation and manipulation
 *  of multi-dimensional arrays for non-critical applications. However, the
 *  arrays are not optimal for heavy calculations, and algorithms critically
 *  dependent on the size and access time of the arrays should not use the
 *  ArrayManager or arrays returned from it. */
template<typename T>
class ArrayManager{
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
	static void createRecursive(Index ranges, void **result);

	/** Recursive helper function for initialized create. */
	static void createRecursive(Index ranges, void **result, T fill);

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
void* ArrayManager<T>::create(const Index &ranges){
	void *result;

	createRecursive(ranges, &result);

	return result;
}

template<typename T>
void ArrayManager<T>::createRecursive(Index ranges, void **result){
	if(ranges.size() == 1){
		*((T**)result) = new T[ranges.at(0)];
	}
	else{
		*((void**)result) = new void*[ranges.at(0)];

		int currentRange = ranges.at(0);
//		ranges.erase(ranges.begin());
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			createRecursive(ranges, &(((void**)(*result))[n]));
	}
}

template<typename T>
void* ArrayManager<T>::create(const Index &ranges, T fill){
	void *result;

	createRecursive(ranges, &result, fill);

	return result;
}

template<typename T>
void ArrayManager<T>::createRecursive(Index ranges, void **result, T fill){
	if(ranges.size() == 1){
		*((T**)result) = new T[ranges.at(0)];
		for(int n = 0; n < ranges.at(0); n++)
			(*((T**)result))[n] = fill;
	}
	else{
		*((void**)result) = new void*[ranges.at(0)];

		int currentRange = ranges.at(0);
//		ranges.erase(ranges.begin());
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			createRecursive(ranges, &(((void**)(*result))[n]), fill);
	}
}

template<typename T>
void ArrayManager<T>::destroy(void *array, const Index &ranges){
	destroyRecursive(array, ranges);
}

template<typename T>
void ArrayManager<T>::destroyRecursive(void *array, Index ranges){
	if(ranges.size() == 1){
		delete [] (T*)array;
	}
	else{
		int currentRange = ranges.at(0);
//		ranges.erase(ranges.begin());
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			destroyRecursive(((void**)array)[n], ranges);

		delete [] (void**)array;
	}
}

template<typename T>
T* ArrayManager<T>::flatten(void *array, const Index &ranges){
	int size = 1;
	for(unsigned int n = 0; n < ranges.size(); n++)
		size *= ranges.at(n);

	T *result = new T[size];

	flattenRecursive(array, ranges, result, 0);

	return result;
}

template<typename T>
void ArrayManager<T>::flattenRecursive(void *array, Index ranges, T *result, int offset){
	if(ranges.size() == 1){
		for(int n = 0; n < ranges.at(0); n++){
			result[offset + n] = ((T*)array)[n];
		}
	}
	else{
		int offsetMultiplier = 1;
		for(int n = 1; n < ranges.size(); n++)
			offsetMultiplier *= ranges.at(n);

		int currentRange = ranges.at(0);
//		ranges.erase(ranges.begin());
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			flattenRecursive(((void**)array)[n], ranges, result, offset + offsetMultiplier*n);
	}
}

template<typename T>
void* ArrayManager<T>::unflatten(T *array, const Index &ranges){
	void *result;

	unflattenRecursive(array, ranges, &result, 0);

	return (void*)result;
}

template<typename T>
void ArrayManager<T>::unflattenRecursive(T *array, Index ranges, void **result, int offset){
	if(ranges.size() == 1){
		*((T**)result) = new T[ranges.at(0)];
		for(int n = 0; n < ranges.at(0); n++)
			(*((T**)result))[n] = array[offset + n];
	}
	else{
		*((void**)result) = new void*[ranges.at(0)];

		int offsetMultiplier = 1;
		for(int n = 1; n < ranges.size(); n++)
			offsetMultiplier *= ranges.at(n);

		int currentRange = ranges.at(0);
//		ranges.erase(ranges.begin());
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			unflattenRecursive(array, ranges, &(((void**)(*result))[n]), offset + offsetMultiplier*n);
	}
}

template<typename T>
void ArrayManager<T>::print(void *array, const Index &ranges){
	printRecursive(array, ranges);
}

template<typename T>
void ArrayManager<T>::printRecursive(void *array, Index ranges){
	if(ranges.size() == 1){
		for(int n = 0; n < ranges.at(0); n++)
			Util::Streams::out << ((T*)array)[n] << "\t";
		Util::Streams::out << "\n";
	}
	else{
		int currentRange = ranges.at(0);
		ranges.popFront();
		for(int n = 0; n < currentRange; n++)
			printRecursive(((void**)array)[n], ranges);
		Util::Streams::out << "\n";
	}
}

};	//End of namespace Util
};	//End of namespace TBTK

#endif
